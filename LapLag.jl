using CSV
using DataFrames
using LinearAlgebra
using NumericalIntegration
using Plots
using Unitful
using UnitfulAstro

u  = Unitful;
ua = UnitfulAstro;
module newUnits
using Unitful
@unit perYear "perYear" perYear 1.0/1u"yr" false
end
u.register(newUnits);


# Populate with planetary parameters
struct Planet
    m::Number       # Mass in Jupiters
    a::Number       # Semi-major axis in au
    e::Number       # Eccentricity
    i::Number       # Inclination in deg
    omega::Number   # Longitude of ascending node in deg
    varpi::Number   # Longitude of pericenter in deg
    n::Number       # Mean motion in deg/year
end


# Populate with Planet structs
mutable struct SolarSystem
    # Bodies in the system; maximum multiplicity 9
    multiplicity::Integer
    star_mass::Number
    p1::Planet
    p2::Planet
    p3::Planet
    p4::Planet
    p5::Planet
    p6::Planet
    p7::Planet
    p8::Planet
    p9::Planet
end


function makeplanet(star_mass::Number, m::Number, a::Number, e::Number,
                    i::Number, omega::Number, varpi::Number)
    m *= ua.Mjup
    a *= ua.AU
    i *= pi/180
    omega *= pi/180
    varpi *= pi/180
    n = ((star_mass / ua.Msun) * ua.GMsun / (a^3))^0.5
    return Planet(m, a, e, i, omega, varpi, n)
end


function makesystem(star_mass::Number, planet_list::Array{Planet})
    null_planet = Planet(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    solsys = SolarSystem(0, 0.0, null_planet, null_planet, null_planet,
                                 null_planet, null_planet, null_planet,
                                 null_planet, null_planet, null_planet)
    multiplicity = size(planet_list)[1]
    setproperty!(solsys, :multiplicity, multiplicity)
    setproperty!(solsys, :star_mass, star_mass)
    for i = 1 : multiplicity
        setproperty!(solsys, fieldnames(SolarSystem)[i+2], planet_list[i])
    end
    return solsys
end


function swapcols(X::AbstractMatrix, i::Integer, j::Integer)
    Y = zeros(size(X))
    for k = 1:size(X, 1)
        Y[k, i], Y[k, j] = X[k, j], X[k, i]
    end
    return Y
end


function laplace_coefficient(j::Integer, s::Float64, alpha::Float64)
    steps = 1_000_000
    psi = collect(range(0.0, pi, length=steps))
    db = cos.(j .* psi) .* (1.0 + alpha^2 .- 2.0 * alpha .* cos.(psi)) .^ (-s)
    b = cumul_integrate(psi, db)
    b *= 2.0/pi
    return b[steps]
end


###################################################################################################
# THE SECULAR MATRICES                                                                            #
###################################################################################################


function ajj(star_mass::Number, planet_list::Array{Planet}, j::Integer)
    multiplicity = size(planet_list)[1]
    terms = zeros(multiplicity) .* ((ua.GMsun)^(1/2) / ua.AU^(3/2))
    jbody = planet_list[j]
    for k = 1 : multiplicity
        if k != j
            term = nothing
            kbody = planet_list[k]
            alpha = jbody.a / kbody.a
            if alpha < 1.0
                term = 0.25 * jbody.n * kbody.m / (star_mass + jbody.m) * alpha^2 * laplace_coefficient(1, 3/2, alpha)
            else
                alpha = kbody.a / jbody.a
                term = 0.25 * jbody.n * kbody.m / (star_mass + jbody.m) * alpha * laplace_coefficient(1, 3/2, alpha)
            end
            terms[k] = term
        end
    end
    component = sum(terms)
    return component * 180.0/pi |> u"perYear"
end


function bjj(star_mass::Number, planet_list::Array{Planet}, j::Integer)
    multiplicity = size(planet_list)[1]
    terms = zeros(multiplicity) * ((ua.GMsun)^(1/2) / ua.AU^(3/2))
    jbody = planet_list[j]
    for k = 1 : multiplicity
        if k != j
            term = nothing
            kbody = planet_list[k]
            alpha = jbody.a / kbody.a
            if alpha < 1.0
                term = -0.25 * jbody.n * kbody.m / (star_mass + jbody.m) * alpha^2 * laplace_coefficient(1, 3/2, alpha)
            else
                alpha = kbody.a / jbody.a
                term = -0.25 * jbody.n * kbody.m / (star_mass + jbody.m) * alpha * laplace_coefficient(1, 3/2, alpha)
            end
            terms[k] = term
        end
    end
    component = sum(terms)
    return component * 180.0/pi |> u"perYear"
end


function ajk(star_mass::Number, planet_list::Array{Planet}, j::Integer, k::Integer)
    jbody = planet_list[j]
    kbody = planet_list[k]
    alpha = jbody.a / kbody.a
    if alpha < 1.0
        component = -0.25 * jbody.n * kbody.m / (star_mass + jbody.m) * alpha^2 * laplace_coefficient(2, 3/2, alpha)
    else
        alpha = kbody.a / jbody.a
        component = -0.25 * jbody.n * kbody.m / (star_mass + jbody.m) * alpha * laplace_coefficient(2, 3/2, alpha)
    end
    return component * 180.0/pi |> u"perYear"
end


function bjk(star_mass::Number, planet_list::Array{Planet}, j::Integer, k::Integer)
    jbody = planet_list[j]
    kbody = planet_list[k]
    alpha = jbody.a / kbody.a
    if alpha < 1.0
        component = 0.25 * jbody.n * kbody.m / (star_mass + jbody.m) * alpha^2 * laplace_coefficient(1, 3/2, alpha)
    else
        alpha = kbody.a / jbody.a
        component = 0.25 * jbody.n * kbody.m / (star_mass + jbody.m) * alpha * laplace_coefficient(1, 3/2, alpha)
    end
    return component * 180.0/pi |> u"perYear"
end


function secular_A(star_mass::Number, planet_list::Array{Planet})
    multiplicity = size(planet_list)[1]
    A = zeros(multiplicity, multiplicity) * u"perYear"
    for j = 1 : multiplicity
        for k = 1 : multiplicity
            if k != j
                A[j, k] = ajk(star_mass, planet_list, j, k)
            else
                A[j, k] = ajj(star_mass, planet_list, j)
            end
        end
    end
    return A
end


function secular_B(star_mass::Number, planet_list::Array{Planet})
    multiplicity = size(planet_list)[1]
    B = zeros(multiplicity, multiplicity) * u"perYear"
    for j = 1 : multiplicity
        for k = 1 : multiplicity
            if k != j
                B[j, k] = bjk(star_mass, planet_list, j, k)
            else
                B[j, k] = bjj(star_mass, planet_list, j)
            end
        end
    end
    return B
end


###################################################################################################
# EQUATIONS OF MOTION                                                                             #
###################################################################################################


function hecc(eccs::Matrix{Float64}, g::Vector{Float64}, beta::Vector{Float64}, t::Float64)
    res = eccs .* sin.(g * t + beta)
    return res
end


function kecc(eccs::Matrix{Float64}, g::Vector{Float64}, beta::Vector{Float64}, t::Float64)
    res = eccs .* cos.(g * t + beta)
    return res
end


function pinc(incs::Matrix{Float64}, f::Vector{Float64}, gamma::Vector{Float64}, t::Float64)
    res = incs .* sin.(f * t + gamma)
    return res
end


function qinc(incs::Matrix{Float64}, f::Vector{Float64}, gamma::Vector{Float64}, t::Float64)
    res = incs .* cos.(f * t + gamma)
    return res
end


function ecc(eccs::Matrix{Float64}, g::Vector{Float64}, beta::Vector{Float64}, t::Float64)
    h = hecc(eccs, g, beta, t)
    k = kecc(eccs, g, beta, t)
    e = sqrt.(h.^2 .+ k.^2)
    return e
end


function inc(incs::Matrix{Float64}, f::Vector{Float64}, gamma::Vector{Float64}, t::Float64)
    p = pinc(incs, f, gamma, t)
    q = qinc(incs, f, gamma, t)
    i = sqrt.(p.^2 .+ q.^2)
    return i
end


###################################################################################################
# SOLVE INITIAL VALUE PROBLEM                                                                     #
###################################################################################################


function get_initial_values(planet_list::Array{Planet})
    multiplicity = size(planet_list)[1]
    # Osculating values of h, k, p, q for each planet
    h = zeros(multiplicity); k = zeros(multiplicity)
    p = zeros(multiplicity); q = zeros(multiplicity)
    for i = 1 : multiplicity
        body = planet_list[i]
        h[i] = body.e * sin(body.varpi)
        k[i] = body.e * cos(body.varpi)
        p[i] = body.i * sin(body.omega)
        q[i] = body.i * cos(body.omega)
    end
    init = Dict{String, Array{Float64}}(
        "h" => h,
        "k" => k,
        "p" => p,
        "q" => q
    )
    return init
end


function get_scales_and_phases(eccs::Matrix{Float64}, incs::Matrix{Float64},
                               planet_list::Array{Planet})
    multiplicity = size(planet_list)[1]
    # Linear algebra that retrieves S_1 * sin(beta_1), T_1 * sin(gamma_1), and so on
    init = get_initial_values(planet_list)
    e_init_matrix = zeros(multiplicity, 2)
    i_init_matrix = zeros(multiplicity, 2)
    for i in 1 : multiplicity
        e_init_matrix[i, 1] = init["h"][i]
        e_init_matrix[i, 2] = init["k"][i]
        i_init_matrix[i, 1] = init["p"][i]
        i_init_matrix[i, 2] = init["q"][i]
    end

    eres = inv(eccs) * e_init_matrix # eres[n, 1] are the S_n*sin(beta_n);
                                     # eres[n, 2] are the S_n*cos(beta_n)
    ires = inv(incs) * i_init_matrix # ires[n, 1] are the T_n*sin(gamma_n);
                                     # ires[n, 2] are the T_n*cos(gamma_n)

    scale_S = zeros(multiplicity)
    scale_T = zeros(multiplicity)
    beta    = zeros(multiplicity)
    gamma   = zeros(multiplicity)
    for i = 1 : multiplicity
        scale_S[i] = sqrt(eres[i, 1]^2 + eres[i, 2]^2)
        scale_T[i] = sqrt(ires[i, 1]^2 + ires[i, 2]^2)
        beta[i]    = atan(eres[i, 1], eres[i, 2])
        gamma[i]   = atan(ires[i, 1], ires[i, 2])
    end

    return scale_S, scale_T, beta, gamma
end


function run_and_plot(time::Vector{Float64},
                      eccs::Matrix{Float64}, incs::Matrix{Float64},
                      g::Vector{Float64}, f::Vector{Float64},
                      beta::Vector{Float64}, gamma::Vector{Float64},
                      multiplicity::Integer)
    # Generate time series
    e_series = [[] for _ in 1 : multiplicity]
    i_series = [[] for _ in 1 : multiplicity]
    for t in time
        es = ecc(eccs, g, beta, t)
        is = inc(incs, f, gamma, t)
        for k = 1 : multiplicity
            push!(e_series[k], es[k])
            push!(i_series[k], is[k])
        end
    end

    # Make plot
    ecc_plot = plot(
        time, e_series,
        xlabel="Time (yr)",
        ylabel="e",
        label=transpose(1:multiplicity),
        linewidth=3
    )
    inc_plot = plot(
        time, i_series,
        xlabel="Time (yr)",
        ylabel="I (deg)",
        label=false,
        linewidth=3
    )
    my_plot = plot(ecc_plot, inc_plot, layout=(1, 2))
    savefig(my_plot, "secular.png")
end


function main()
    ## TEST SYSTEM
    # Sun
    msun = 1.0 * ua.Msun
    # Jupiter
    jupiter = makeplanet(msun, 1.0, 5.202545, 0.0474622, 1.30667, 100.0381, 13.983865)
    # Saturn
    msat = 2.85837/9.54786 # Mass in Mjup
    saturn = makeplanet(msun, msat, 9.554841, 0.0575481, 2.48795, 113.1334, 88.719425)
    # 2-planet system
    planet_list = [jupiter, saturn]
    solsys = makesystem(msun, planet_list) # use the SolarSystem struct for everything if all else works!

    tt = collect(range(-1.0e5, 1.0e5, 10_000))

    A = secular_A(solsys.star_mass, planet_list)
    B = secular_B(solsys.star_mass, planet_list)

    g = eigvals(A / 1u"perYear")
    f = eigvals(B / 1u"perYear")
    eccs = eigvecs(A / 1u"perYear")
    incs = eigvecs(B / 1u"perYear")
    incs = swapcols(incs, 1, 2)

    scale_S, scale_T, beta, gamma = get_scales_and_phases(eccs, incs, planet_list)
    eccs_scaled = Matrix(transpose(scale_S) * eccs)
    incs_scaled = Matrix(transpose(scale_T) * incs)

    run_and_plot(tt, eccs_scaled, incs_scaled, g, f, beta, gamma, solsys.multiplicity)
end


main()
