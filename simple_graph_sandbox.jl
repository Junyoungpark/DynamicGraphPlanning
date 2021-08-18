using LightGraphs
using GraphPlot
using Random, Colors

Random.seed!(42)
using BenchmarkTools: @btime
using StatProfilerHTML

using MetaGraphs, SimpleWeightedGraphs
using SNAPDatasets
# https://snap.stanford.edu/data/index.html
using StatsBase: sample, Weights

using POMDPs
using Parameters
using MCTS

# ---- Definitions

@enum NodeState S=1 E=2 I=3 R=4
@enum Measure business_as_usual=0 mask_mandate=1 stay_at_home_order=2

# struct GraphState
#     graph::SimpleWeightedGraph
#     state::Vector{NodeState}
#     function GraphState(g::SimpleWeightedGraph, s::Vector{NodeState})
#         @assert nv(wg) == length(s)
#         return new(g, s)
#     end
# end

struct GraphState
    graph::MetaGraph
    state::Vector{NodeState}
    function GraphState(g::MetaGraph, s::Vector{NodeState})
        @assert nv(wg) == length(s)
        return new(g, s)
    end
end
GraphAction = Vector{Measure}

@with_kw struct SEIRConfiguration
    β::Float64 = 0.6  # catch it if exposed (S -> E)
    σ::Float64 = 0.8  # develop symptoms rate / become infectious (E -> I)
    η::Float64 = 0.4  # recovery rate (I -> R)
    ξ::Float64 = 0.05 # immunity wear-off rate (R -> S)
end

@with_kw struct GraphMDP <: MDP{GraphState, GraphAction}
    _seir_dynamics::SEIRConfiguration = SEIRConfiguration()
    γ::Float64 = 0.9
    node_affiliation::Vector{Int64}
end

# ---- Methods

function set_weight(g::SimpleWeightedGraph, src::Int64, dst::Int64, w::Float64)
    if (adjacency_matrix(g)[src, dst] == 0) return false end
    weights(g)[src, dst] = w
    weights(g)[dst, src] = w
    # when setting the adjacency matrix on simple weighted graphs the information does not stay
    return true
end
set_weight(g::GraphState, src::Int64, dst::Int64, w::Float64) = set_weight(g.graph, src, dst, w)
set_state(g::GraphState, n::Int64, s::NodeState) = g.state[n] = s

function get_affected_nodes(affiliation::Vector{Int64}, a::GraphAction, m::Measure)
    communities = findall(x->x==m, a)
    return findall(x -> x in communities, affiliation)
end

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
cost(m::Measure) = m == business_as_usual ? 0.0 : (m == mask_mandate ? 1.0 : 3.0)
inter_score(m1::Measure, m2::Measure) = max(score(m1), score(m2))

function transition(m::GraphMDP, s::GraphState, a::GraphAction)
    # SEIR model
    gp = deepcopy(s.graph)
    sp = deepcopy(s.state)
    
    # apply community measures
    for e in edges(gp)
        set_prop!(gp, e.src, e.dst, :weight, 
            inter_score(a[m.node_affiliation[e.src]], a[m.node_affiliation[e.dst]]))
        # set_weight(sp, e.src, e.dst, inter_score(a[m.node_affiliation[e.src]], a[m.node_affiliation[e.dst]]))
    end
    
    # who is exposed
    infected_nodes = findall(x->x==I, sp)
    for v in findall(x->x==S, sp)
        # prob of catching it (given n infected neighbors)
        # = 1 - prob of not catching it from any of them
        # = 1 - ∏^n (1 - p^transmittion_n)
        infected_neigh = findall(in(infected_nodes), neighbors(gp, v))
        if !isempty(infected_neigh) && rand() > prod(1 .-weights(gp)[v, infected_neigh]) 
            sp[v] = E
        end
    end
    
    # transition graph temporally
    sp = broadcast(z -> seir(m._seir_dynamics, z), sp)
    return GraphState(gp, sp)
end

function reward(m::GraphMDP, s::GraphState, a::GraphAction)
    return sum([length(get_affected_nodes(m.node_affiliation, a, meas)) * cost(meas) for meas in instances(Measure)])
end

get_action_space(m::GraphMDP) = Iterators.product(fill(instances(Measure), n_communities(m))...) .|> collect |> vec
POMDPs.actions(m::GraphMDP) = a_space_cache
POMDPs.discount(m::GraphMDP) = m.γ

function POMDPs.gen(m::GraphMDP, s::GraphState, a::GraphAction, rng = MersenneTwister())
    return (sp = transition(m,s,a), r = reward(m,s,a))
end

# ---- MAIN

# wg = SimpleWeightedGraph(loadsnap(:facebook_combined))
wg = MetaGraph(loadsnap(:facebook_combined))
states = sample([S, I], Weights([0.9, 0.1]), nv(wg))
s = GraphState(wg, states)

labels = sample(1:12, nv(wg))
mdp = GraphMDP(node_affiliation=labels)
a_space_cache = get_action_space(mdp)

solver = MCTSSolver(n_iterations=5, depth=10, exploration_constant=1.0)
planner = solve(solver, mdp)
;

# a = rand(a_space_cache)
# gen(mdp, s, a)

# a = action(planner, mg)
