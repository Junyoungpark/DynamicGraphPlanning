using LightGraphs
using GraphPlot
using Random
using StatsBase: sample

Random.seed!(42)

using MetaGraphs
using SNAPDatasets
# https://snap.stanford.edu/data/index.html
g = loadsnap(:facebook_combined) 
labels = sample(1:16, nv(g))

# defintions
@enum NodeState S=1 E=2 I=3 R=4
@enum Measure business_as_usual=0 mask_mandate=1 stay_at_home_order=2

mg = MetaGraph(g, 0.6)
for (i, label) in enumerate(labels)
    set_prop!(mg, i, :label, label)
    set_prop!(mg, i, :state, rand() < 0.05 ? I : S) # 5% infection rate
end

using POMDPs
using Parameters

GraphState = typeof(mg)
GraphAction = Vector{Measure}

@with_kw struct SEIRConfiguration
    β::Float64 = 0.6 # catch it if exposed (S -> E)
    σ::Float64 = 0.8 # develop symptoms rate / become infectious (E -> I)
    η::Float64 = 0.4 # recovery rate (I -> R)
    ξ::Float64 = 0.05# immunity wear-off rate (R -> S)
end

@with_kw struct GraphMDP <: MDP{GraphState, GraphAction}
    _seir_dynamics::SEIRConfiguration = SEIRConfiguration()
    γ::Float64 = 0.9
    node_affiliation::Vector{Int64}
end

function get_affected_nodes(affiliation::Vector{Int64}, a::GraphAction, m::Measure)
    communities = findall(x->x==m, a)
    return findall(x -> x in communities, affiliation)
end
n_communities(m::GraphMDP) = length(unique(m.node_affiliation))
score(m::Measure) = m == business_as_usual ? 0.6 : (m == mask_mandate ? 0.2 : 0.05)
cost(m::Measure) = m == business_as_usual ? 0.0 : (m == mask_mandate ? 1.0 : 3.0)
inter_score(m1::Measure, m2::Measure) = max(score(m1), score(m2))

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

function transition(m::GraphMDP, s::GraphState, a::GraphAction)
    # SEIR model
    sp = copy(s)
    
    # apply community measures
    for e in edges(sp)
        set_prop!(sp, e.src, e.dst, :weight, 
            inter_score(a[m.node_affiliation[e.src]], a[m.node_affiliation[e.dst]]))
    end
    
    # who is exposed
    infected_nodes = collect(filter_vertices(s, :state, I))
    filter_vertices(s, :state, S) .|> 
        v -> rand() > prod(weights(s)[v, findall(in(infected_nodes), neighbors(s, v))]) ? 
        nothing : set_prop!(s, v, :state, E)
    
    # transition graph temporally
    vertices(sp) .|> x -> set_prop!(sp, x, :state, seir(m._seir_dynamics , get_prop(sp, x, :state)))
    return sp
end

function reward(m::GraphMDP, s, a)
    return sum([length(get_affected_nodes(m.node_affiliation, a, meas)) * cost(meas) for meas in instances(Measure)])
end

function POMDPs.actions(m::GraphMDP)
#     return Iterators.product(fill(instances(Measure), n_communities(m))...) .|> collect |> vec
    return a_space_cache
end

function POMDPs.gen(m::GraphMDP, s::GraphState, a::GraphAction, rng = MersenneTwister())
    return (sp = transition(m,s,a), r = reward(m,s,a))
end

POMDPs.discount(m::GraphMDP) = m.γ 

# Test it

using MCTS
using StatProfilerHTML

mdp = GraphMDP(node_affiliation=labels);
global a_space_cache = Iterators.product(fill(instances(Measure), n_communities(mdp))...) .|> collect |> vec
solver = MCTSSolver(n_iterations=50, depth=10, exploration_constant=5.0)
planner = solve(solver, mdp)
;
# @profilehtml action(planner, mg)