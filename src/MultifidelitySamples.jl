module MultifidelitySamples

using Distributions, Distributed

export AbstractLikelihood, PlaceholderLikelihood
export MultifidelityPathLikelihood
export MultifidelityPathTree, MultifidelitySample

export evaluate, burden
export weighted_mean
export cost_functional


"""
    AbstractLikelihood

An abstract type corresponding to a *monofidelity* likelihood.
Concrete subtypes `L <: AbstractLikelihood` need to implement `evaluate(::L, θ)`.
Optionally, they need to also implement `evaluate(::L, θ, y...)` to allow for coupling between (likelihood-free) likelihoods.
"""
abstract type AbstractLikelihood end

"""
    evaluate(::AbstractLikelihood, θ)

Define for concrete subtypes of `AbstractLikelihood`.
Return `(ω::Float64, v::Vector{Float64})` where
- `ω::Float64` represents the likelihood (or likelihood-free estimate of the likelihood)
- `v::Vector{Float64}` can be used for coupling between likelihoods and in any mean function `μ` used to define a `MultifidelityPathLikelihood`
"""
function evaluate(::AbstractLikelihood, θ) end
evaluate(L::AbstractLikelihood, θ, parent, ancestors...) = evaluate(L, θ, ancestors...)


"""
    PlaceholderLikelihood(ω::Float64) <: AbstractLikelihood

A concrete likelihood to enable quick checking that code compiles and executes.
The likelihood is specified as `ω`.
The 
"""
struct PlaceholderLikelihood <: AbstractLikelihood
    ω::Float64
end
function evaluate(L::PlaceholderLikelihood, θ::Real)
    sleep(0.0001 * θ)
    L.ω, [L.ω]
end


"""
    MultifidelityPathLikelihood{N, LO <: NTuple{N, AbstractLikelihood}, HI <: AbstractLikelihood, M}

Contains
- `L_hi::HI` as high-fidelity likelihood.
- `L_lo::LO` as a tuple (i.e. ordered list) of low-fidelity likelihoods to be sequentially evaluated as approximations of `L_hi`.
- `μ::M` as a function determining the (Poisson) mean of evaluations of each likelihood.
For `LVL` in `0:(N-1)` the function `μ(θ, y::Vararg{Vector{Float64}, LVL})` determines the Poisson mean of how many evaluations of `L_lo[LVL+1]` we make, conditioned on all preceding evaluations and `θ`
At the final level, the function `μ(θ, y::Vararg{Vector{Float64}, N})` determines the Poisson mean of how many evaluations of 'L_hi' we make, conditioned on all preceding evaluations and `θ`.
"""
struct MultifidelityPathLikelihood{N, LO<:NTuple{N, AbstractLikelihood}, HI<:AbstractLikelihood, M}
    L_hi::HI
    L_lo::LO
    μ::M
end
function MultifidelityPathLikelihood(L_hi::AbstractLikelihood, L_lo::AbstractLikelihood...; μ = (θ, y...) -> one(Float64))
    MultifidelityPathLikelihood(L_hi, L_lo, μ)
end
function Base.show(io::IO, L_ρ::MultifidelityPathLikelihood{N}) where N
    println(io, "MultifidelityPathLikelihood:")
    for n in 1:N
        println(io, "L_{lo, ", n, "}: ", L_ρ.L_lo[n])
    end
    println(io, "L_hi: ", L_ρ.L_hi)
end


"""
    MultifidelityPathTree{LVL, Θ, T<:MultifidelityPathTree{LVL+1, Θ}}

Contains
- `ω::Float64` as the evaluated likelihood at level `LVL`
- `z::Tuple{Θ, Vararg{Vector{Float64}, LVL}}` as the accumulated information from this and preceding levels, and `θ`.
- `c::Float64` as the computational burden of evaluating the likelihood `ω` (i.e. the simulation burden, for likelihood-free inference).
- `μ::Float64` as the computed mean for the Poisson number of correction trees to generate.
- `children::T` as a vector of correction trees, rooted at the next level.
"""
struct MultifidelityPathTree{LVL, Θ, T}
    ω::Float64
    z::Tuple{Θ, Vararg{Vector{Float64}, LVL}}
    c::Float64
    μ::Float64
    children::Vector{T}
    function MultifidelityPathTree(ω, z::Tuple{Θ, Vararg{Vector{Float64}, LVL}}, c, μ, children::Vector{T}) where {LVL, Θ, T}
        T <: MultifidelityPathTree{LVL+1, Θ} || error("Level mismatch")
        new{LVL, Θ, T}(ω, z, c, μ, children)
    end
end
function Base.show(io::IO, t::MultifidelityPathTree)
    sep = ", "
    print(io, "MultifidelityPathTree(", t.ω, sep, t.z, sep, t.c, sep, t.μ, sep, length(t.children), " children)")
end


"""
    get_θ(T::MultifidelityPathTree)

Return the parameter value of the tree `T`.
"""
get_θ(T::MultifidelityPathTree) = T.z[1]


"""
    MultifidelitySample{Θ, T <: MultifidelityPathTree{1, Θ}}

A vector of trees rooted at level zero: `Vector{MultifidelityPathTree{0, Θ, T}}`
"""
MultifidelitySample{Θ, T <: MultifidelityPathTree{1, Θ}} = Vector{MultifidelityPathTree{0, Θ, T}}


"""
    evaluate(T::MultifidelityPathTree{0})

Overloaded `evaluate` to read off the multifidelity likelihood from the tree `T` rooted at level 0.
"""
evaluate(T::MultifidelityPathTree{0}) = _evaluate(T)
function _evaluate(T::MultifidelityPathTree)
    ω = T.ω
    for twig in T.children
        ω += (_evaluate(twig) - T.ω) / T.μ
    end
    return ω
end


"""
    MultifidelityPathTree(L_ρ::MultifidelityPathLikelihood{N}, ω, c, θ::Θ, y::Vararg{Vector{Float64}, LVL}) where {Θ, N, LVL}

Returns a `MultifidelityPathTree` rooted at level `LVL`.
We wrap `ω` and `c` and `z = (θ, y...)` in this tree.
We use `z` as the argument to `L_ρ.μ` to calculate the mean number, `μ`, of correction trees to generate (which gets wrapped also).
Each tree at the next level is generated in turn and stored in `children`, before the completed tree is returned.
"""
function MultifidelityPathTree(L_ρ::MultifidelityPathLikelihood{N}, ω, c, θ::Θ, ancestors::Vararg{Vector{Float64}, LVL}) where {Θ, N, LVL}
    z = (θ, ancestors...)
    μ = LVL > N ? 0.0 : L_ρ.μ(z...)
    tree = MultifidelityPathTree(ω, z, c, μ, MultifidelityPathTree{LVL+1, Θ}[])
    
    M::Integer = LVL > N ? 0 : rand(Poisson(μ))
    if M > 0
        L_next = LVL==N ? L_ρ.L_hi : L_ρ.L_lo[LVL+1]
        for _m in 1:M
            timed_likelihood = @timed evaluate(L_next, θ, ancestors...)
            ω_m, y_m = timed_likelihood.value
            c_m = timed_likelihood.time
            push!(tree.children, MultifidelityPathTree(L_ρ, ω_m, c_m, θ, y_m, ancestors...))
        end
    end

    return tree
end

"""
    MultifidelitySample(L_ρ::MultifidelityPathLikelihood, prior, N::Integer)

Returns `S::MultifidelitySample` as a vector of `MultifidelityPathTree{0}` of `length(S) = N`.
Requires `rand(prior)` to be implemented to generate `θ` values, the production of which is timed.

Note: Uses `pmap` if possible and warns if this is not exploited.
"""
function MultifidelitySample(L_ρ::MultifidelityPathLikelihood, prior, N::Integer)
    n = nworkers()
    @info (isone(n) ? "Only 1 worker: look at parallelism." : "$(nworkers()) workers!")
    S = pmap(1:N) do _i
        timed_prior_sample = @timed rand(prior)
        c_0 = timed_prior_sample.time
        θ_i = timed_prior_sample.value
        T_i = MultifidelityPathTree(L_ρ, 0.0, c_0, θ_i)
    end
    return S
end


"""
    weighted_mean(G, S::MultifidelitySample)

Returns the Monte Carlo mean of the function `G(θ)`, based on the multifidelity weights evaluated in each element of `S::MultifidelitySample`.
"""
function weighted_mean(G, S::MultifidelitySample)
    num = 0.0
    den = 0.0
    for T in S
        ω = evaluate(T)
        Gθ = G(get_θ(T))
        num += ω*Gθ
        den += ω
    end
    return num/den
end


"""
    burden(T::MultifidelityPathTree)

Returns the total observed computational burden for each node of the tree `T`.
"""
function burden(T::MultifidelityPathTree)
    c = T.c
    for twig in T.children
        c += burden(twig)
    end
    return c
end

"""
    burden(S::MultifidelitySample)

Returns the total observed computational burden of the Monte Carlo sample `S` of multifidelity trees.
"""
burden(S::MultifidelitySample) = sum(burden, S)

function _burden_functional(μ) 
    function _burden(T::MultifidelityPathTree)
        c = 0.0
        for twig in T.children
            c += _burden(twig) / T.μ
        end
        c *= μ(T.z...)
        c += T.c
        return c
    end
    _burden(S::MultifidelitySample) = mean(_burden, S)
    return _burden
end

function _variance_functional(μ, G)
    function _variance(T::MultifidelityPathTree, x)
        numChildren = length(T.children)
        if iszero(numChildren)
            return 0.0
        else
            v = 0.0
            for i in 1:numChildren
                v += _variance(T.children[i], T.ω)
            end
            v *= T.μ / μ(T.z...)
            for i in 1:numChildren
                Ω_i = _evaluate(T.children[i])
                for j in (i+1):numChildren
                    Ω_j = _evaluate(T.children[j])
                    v += 2 * (Ω_i - x) * (Ω_j - x)
                end
            end
            v /= T.μ
            v /= T.μ
            return v
        end
    end
    function _variance(S::MultifidelitySample)
        Gbar = weighted_mean(G, S)
        Δ²(θ) = abs2(G(θ) - Gbar)
        Δ²(T::MultifidelityPathTree{0}) = Δ²(get_θ(T))
        Δ²V(T) = Δ²(T)*_variance(T, 0.0)
        return mean(Δ²V, S)
    end
    return _variance
end


"""
    cost_functional(
        μ = (θ, y...) -> mean_fun(θ, y...),
        G = θ -> estimate_fun(θ)
    )

Returns the function `J(S::MultifidelitySample)::Float64` which is used as a Monte Carlo estimate of the product of
- the mean simulation time per iteration
- the mean squared error proxy.

Given `G` and `S`, we aim to use `Flux.jl` to optimise the function `μ`, with `J(S)` as the cost function that evolves with `mu`.
"""
function cost_functional(μ, G)
    _burden = _burden_functional(μ)
    _variance = _variance_functional(μ, G)
    _cost(S::MultifidelitySample) = _burden(S) * _variance(S)
    return _cost
end

end # module