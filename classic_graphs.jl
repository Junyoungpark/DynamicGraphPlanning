using LightGraphs
using LightGraphs: src, dst, edgetype, vertices, edges, add_vertex!, rem_vertex!, has_vertex, has_edge, inneighbors, outneighbors, indegree, outdegree, degree, induced_subgraph, SimpleGraphEdge, SimpleEdge, reverse
import LightGraphs: nv, ne, is_directed, add_edge!, rem_edge!, edges, vertices, neighbors, weights

using Random
using StatsBase: sample
Random.seed!(42)

mutable struct ClassicGraph{T <: Integer, U <: Real, S} #<: AbstractClassicGraph{T}
    graph::SimpleGraph{T}
    # edges
    weights::Vector{Vector{U}} # [src]: (dst, dst, dst)
    defaultweight::U
    # nodes
    states::Vector{S}
end


function ClassicGraph(g::SimpleGraph, vs::Vector{S}, defaultweight::U = 1.0) where U <: Real where S
    T = eltype(g)

    @assert length(vs) == nv(g)
    ws = [[defaultweight for _ in src] for src in g.fadjlist]
    
    ClassicGraph(g, ws, defaultweight, vs)
end
ClassicGraph(x, vs::Vector{S}, defaultweight::U = 1.0) where U <: Real where S = ClassicGraph(SimpleGraph(x), vs, defaultweight)
ClassicGraph(g::SimpleGraph, f, defaultweight::U = 1.0) where U <: Real where S = ClassicGraph(g, vertices(g) |> f, defaultweight)
ClassicGraph(x, f, defaultweight::U = 1.0) where U <: Real where S = ClassicGraph(SimpleGraph(x), f, defaultweight)

nv(g::ClassicGraph) = nv(g.graph)
ne(g::ClassicGraph) = ne(g.graph)
edges(g::ClassicGraph) = edges(g.graph)
vertices(g::ClassicGraph) = vertices(g.graph)
neighbors(g::ClassicGraph, v) = neighbors(g.graph, v)

function ClassicGraph(g::ClassicGraph{T, U, S}) where T <: Integer where U <: Real where S
    return ClassicGraph(deepcopy(g.graph), deepcopy(g.weights), g.defaultweight, deepcopy(g.states))
end

SimpleGraph(g::ClassicGraph) = g.graph

is_directed(::Type{ClassicGraph}) = false
is_directed(::Type{ClassicGraph{T,U,S}}) where T where U where S = false
is_directed(g::ClassicGraph) = false

# indexing
function get_index(g::ClassicGraph{T,U,S}, _e::SimpleEdge) where T where U where S
    s, d = T.(Tuple(_e))
    verts = vertices(g.graph)
    (s in verts && d in verts) || return false  # edge out of bounds
    @inbounds list = g.graph.fadjlist[s]
    index = findfirst(x->x==d, list)
    return isnothing(index) ?  -1 : index # return -1 if edge does not exist
end
get_dst_index(g, _e) = get_index(g, _e)
get_src_index(g, _e) = get_index(g, reverse(_e))

# weights
weighttype(g::ClassicGraph{T,U,S}) where T where U where S = U
function get_weight(g::ClassicGraph, _e::SimpleEdge)
    idx = get_index(g, _e)
    return idx > 0 ? g.weights[_e.src][idx] : 0.0
end
function set_weight!(g::ClassicGraph, _e::SimpleEdge, w::U) where U
    idx = get_dst_index(g, _e)
    idx > 0 || return false
    g.weights[_e.src][idx] = w

    idx = get_src_index(g, _e)
    idx > 0 || return false # this should never happen
    g.weights[_e.dst][idx] = w
    return true
end

adjacency_matrix(g::ClassicGraph) = adjacency_matrix(g.graph)
function weights(g::ClassicGraph)
    w_adj = zeros(nv(g), nv(g))
    range(1, stop=nv(g)) .|> i -> w_adj[i, g.graph.fadjlist[i]] = g.weights[i]
    return w_adj
end

# states
statetype(g::ClassicGraph{T,U,S}) where T where U where S = S
get_state(g::ClassicGraph, _v::Integer) = g.states[_v]
function set_state!(g::ClassicGraph, _v::Integer, s::S) where S
    _v <= length(g.states) || return false
    g.states[_v] = s
    return true
end

# add / remove edges
"""
    add_edge!(g, e)
Add an edge `e` to graph `g`. Return `true` if edge was added successfully,
otherwise return `false`.
"""
function add_edge!(g::ClassicGraph{T,U,S}, _e::SimpleGraphEdge{T}, w::U) where T where U where S
    add_edge!(g.graph, _e) || return false # edge out of bounds OR edge already in graph

    idx = get_dst_index(g, _e)
    idx > 0 || return false
    insert!(g.weights[_e.src], idx, w)

    idx = get_src_index(g, _e)
    idx > 0 || return false # this should never happen
    insert!(g.weights[_e.dst], idx, w)
    return true
end
add_edge!(g::ClassicGraph, _e::SimpleGraphEdge) = add_edge!(g, _e, d.defaultweight)

"""
    rem_edge!(g, e)
Remove an edge `e` from graph `g`. Return `true` if edge was removed successfully,
otherwise return `false`.
### Implementation Notes
If `rem_edge!` returns `false`, the graph may be in an indeterminate state, as
there are multiple points where the function can exit with `false`.
"""
function rem_edge!(g::ClassicGraph{T,U,S}, _e::SimpleGraphEdge{T}) where T where U where S
    idx = get_dst_index(g, _e)
    idx > 0 || return false
    deleteat!(g.weights[_e.src], idx)

    idx = get_src_index(g, _e)
    idx > 0 || return false # this should never happen
    deleteat!(g.weights[_e.dst], idx)
    
    return rem_edge!(g.graph, _e)
end

## TODO : support add / rem vertices

;

