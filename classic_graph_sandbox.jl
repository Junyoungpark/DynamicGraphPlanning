include("classic_graphs.jl")
using GraphPlot
using SNAPDatasets
# https://snap.stanford.edu/data/index.html

using Random, Colors
using StatsBase: sample, Weights
Random.seed!(42)

using BenchmarkTools: @btime
using StatProfilerHTML

using POMDPs, POMDPSimulators
using Parameters
using MCTS

using Clustering

# ---- Definitions

@enum NodeState S=1 E=2 I=3 R=4
@enum Measure business_as_usual=0 mask_mandate=1 stay_at_home_order=2

GraphState = ClassicGraph
GraphAction = Vector{Measure}

@with_kw struct SEIRConfiguration
    β::Float64 = 0.6  # catch it if exposed (S -> E)
    σ::Float64 = 0.8  # develop symptoms rate / become infectious (E -> I)
    η::Float64 = 0.1  # recovery rate (I -> R)
    ξ::Float64 = 0.05 # immunity wear-off rate (R -> S)
end

@with_kw struct GraphMDP <: MDP{GraphState, GraphAction}
    _seir_dynamics::SEIRConfiguration = SEIRConfiguration()
    γ::Float64 = 0.9
    node_affiliation::Vector{Int64}
end

# ---- Methods

function get_affected_nodes(affiliation::Vector{Int64}, a::GraphAction, m::Measure)
    communities = findall(x->x==m, a)
    return findall(x -> x in communities, affiliation)
end
fractionHealthy(s::GraphState) = (sum(s.states .== S) + sum(s.states .== R)) / nv(s)
get_breakdown(s::GraphState) = (sum(s.states .== S), sum(s.states .== E), sum(s.states .== I), sum(s.states .== R))

function seir(config::SEIRConfiguration, s::NodeState)
    if s == S
        return S
    elseif s == E
        return rand() < config.σ ? I : E
    elseif s == I 
        return rand() < config.η ? R : I
    end
    return rand() < config.ξ ? S : R
end

n_communities(m::GraphMDP) = length(unique(m.node_affiliation))
score(m::Measure) = m == business_as_usual ? 0.6 : (m == mask_mandate ? 0.2 : 0.05)
cost(m::Measure) = m == business_as_usual ? 0.0 : (m == mask_mandate ? -1.0 : -3.0)
inter_score(m1::Measure, m2::Measure) = max(score(m1), score(m2))

function transition(m::GraphMDP, s::GraphState, a::GraphAction)
    # SEIR model
    sp = ClassicGraph(s)
    
    # apply community measures
    edges(sp) .|> e -> set_weight!(sp, e, inter_score(a[m.node_affiliation[e.src]], a[m.node_affiliation[e.dst]]))
    
    # who is exposed
    infected_nodes = findall(x->x==I, sp.states)
    W = weights(sp)
    for v in findall(x->x==S, sp.states)
        # prob of catching it (given n infected neighbors)
        # = 1 - prob of not catching it from any of them
        # = 1 - ∏^n (1 - p^transmittion_n)
        infected_neigh = neighbors(sp, v)[findall(in(infected_nodes), neighbors(sp, v))]
        if !isempty(infected_neigh)
            # println("# of infected neighbors $(length(infected_neigh)) -> p = $(prod(1 .- W[v, infected_neigh]))")
            if rand() > prod(1 .- W[v, infected_neigh]) 
                set_state!(sp, v, E)
            end
        end
    end
    
    # transition graph temporally
    vertices(sp) .|> z -> set_state!(sp, z, seir(m._seir_dynamics, get_state(sp, z)))
    return sp
end

function reward(m::GraphMDP, s::GraphState, a::GraphAction)
    meas_cost = sum([length(get_affected_nodes(m.node_affiliation, a, meas)) * cost(meas) for meas in instances(Measure)])
    return meas_cost - 0.1 * sum(s.states .== I)
end

get_action_space(m::GraphMDP) = Iterators.product(fill(instances(Measure), n_communities(m))...) .|> collect |> vec
POMDPs.actions(m::GraphMDP) = a_space_cache
POMDPs.discount(m::GraphMDP) = m.γ

function POMDPs.gen(m::GraphMDP, s::GraphState, a::GraphAction, rng = MersenneTwister())
    return (sp = transition(m,s,a), r = reward(m,s,a))
end

function cluster(s::GraphState, k::Int64, linkage::Symbol = :complete)
    println("getting distance matrix...")
    D = distance_matrix(s)
    println("... & clustering into ", k)
    dendogram = hclust(D, linkage=linkage)
    return cutree(dendogram, k = k)
end

struct constAction <: Policy
    _a::GraphAction
end
POMDPs.action(policy::constAction, s::GraphState) = policy._a

# ---- MAIN

g = loadsnap(:facebook_combined)
states = sample([S, I], Weights([0.8, 0.2]), nv(g))
s_i = ClassicGraph(g, states, 0.6)

labels = cluster(s_i, 10)
mdp = GraphMDP(node_affiliation = labels)
a_space_cache = get_action_space(mdp)

POMDPs.initialstate(mdp::GraphMDP) = s_i

# solver = MCTSSolver(n_iterations=20, depth=5, exploration_constant=1.0)
# planner = solve(solver, mdp)

# add rollout policy

doNothing() = constAction(a_space_cache[1])
policy = doNothing()
for (s, a, r) in stepthrough(mdp, policy, s_i, "s,a,r", max_steps=25)
    println("in state $(get_breakdown(s))")
    # println("took action $a")
    println("received reward $r\n")
end