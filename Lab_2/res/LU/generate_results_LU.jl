include(joinpath(@__DIR__, "..", "..", "src", "LU_factorization.jl"))
include(joinpath(@__DIR__, "..", "..", "lib", "strassen_padded.jl"))

using Random
using Statistics
using Printf
using Dates
using Plots
using .LU_factorization
using .strassen_padded

default_mul(A, B, add_count::Ref{Int}=Ref(0), mul_count::Ref{Int}=Ref(0)) = (A * B, add_count[], mul_count[])

const CONFIG = (
    max_size = 1040,
    step = 10,
    trials = 10,
    random_seed = 2025,
    output_dir = joinpath(@__DIR__, "out")
)

function generate_matrix(n::Int)
    M = rand(n, n)
    A = M * transpose(M)
    max_val = maximum(A)
    A .= 0.9999999 .* (A ./ max_val) .+ 1e-8
    return A
end

function build_sizes(max_size::Int, step::Int)
    sizes = Int[]
    value = 1
    while value <= max_size
        push!(sizes, value)
        value += step
    end
    if sizes[end] != max_size
        push!(sizes, max_size)
    end
    return unique(sizes)
end

function summarize(records, sizes)
    grouped = Dict{Int,Vector{NamedTuple{(:size,:trial,:adds,:muls,:time),Tuple{Int,Int,Int,Int,Float64}}}}()
    for rec in records
        push!(get!(grouped, rec.size, NamedTuple{(:size,:trial,:adds,:muls,:time),Tuple{Int,Int,Int,Int,Float64}}[]), rec)
    end
    summary = Dict{Int,NamedTuple{(:avg_adds,:avg_muls,:avg_time),Tuple{Float64,Float64,Float64}}}()
    for n in sizes
        subset = grouped[n]
        adds = [r.adds for r in subset]
        muls = [r.muls for r in subset]
        times = [r.time for r in subset]
        summary[n] = (avg_adds = mean(adds), avg_muls = mean(muls), avg_time = mean(times))
    end
    return summary
end

function persist_results(records, summary; output_dir::AbstractString, max_size::Int, step::Int, trials::Int, random_seed::Int)
    mkpath(output_dir)
    raw_path = joinpath(output_dir, "lu_factorization_raw.csv")
    open(raw_path, "w") do io
        println(io, "size,trial,adds,muls,time_seconds")
        for r in records
            println(io, "$(r.size),$(r.trial),$(r.adds),$(r.muls),$(r.time)")
        end
    end
    summary_path = joinpath(output_dir, "lu_factorization_summary.csv")
    open(summary_path, "w") do io
        println(io, "size,avg_adds,avg_muls,avg_time_seconds")
        for n in sort(collect(keys(summary)))
            stats = summary[n]
            println(io, "$(n),$(stats.avg_adds),$(stats.avg_muls),$(stats.avg_time)")
        end
    end
    config_path = joinpath(output_dir, "lu_factorization_config.txt")
    open(config_path, "w") do io
        @printf(io, "max_size=%d\nstep=%d\ntrials=%d\nrandom_seed=%d\n", max_size, step, trials, random_seed)
    end
    return (raw = raw_path, summary = summary_path, config = config_path)
end

function plot_results(summary, sizes; output_dir::AbstractString)
    sizes_sorted = sort(sizes)
    avg_adds = [summary[n].avg_adds for n in sizes_sorted]
    avg_muls = [summary[n].avg_muls for n in sizes_sorted]
    avg_times = [summary[n].avg_time for n in sizes_sorted]

    p_ops = plot(
        sizes_sorted,
        avg_adds;
        label = "Additions",
        xlabel = "Matrix size (n)",
        ylabel = "Average count",
        title = "Average operation counts vs size (LU)",
        marker = :circle,
        legend = :topleft,
        linewidth = 2
    )
    plot!(p_ops, sizes_sorted, avg_muls; label = "Multiplications", marker = :diamond, linewidth = 2)

    safe_adds = [max(val, eps(Float64)) for val in avg_adds]
    safe_muls = [max(val, eps(Float64)) for val in avg_muls]
    safe_times = [max(val, eps(Float64)) for val in avg_times]

    p_ops_log = plot(
        sizes_sorted,
        safe_adds;
        label = "Additions",
        xlabel = "Matrix size (n)",
        ylabel = "Average count (log scale)",
        title = "Average operation counts vs size (LU, log-log)",
        marker = :circle,
        legend = :topleft,
        linewidth = 2,
        yscale = :log10
    )
    plot!(p_ops_log, sizes_sorted, safe_muls; label = "Multiplications", marker = :diamond, linewidth = 2)

    p_time = plot(
        sizes_sorted,
        avg_times;
        label = "Runtime",
        xlabel = "Matrix size (n)",
        ylabel = "Average time (s)",
        title = "Average runtime vs size (LU)",
        marker = :circle,
        legend = :topleft,
        linewidth = 2
    )

    p_time_log = plot(
        sizes_sorted,
        safe_times;
        label = "Runtime",
        xlabel = "Matrix size (n)",
        ylabel = "Average time (s, log scale)",
        title = "Average runtime vs size (LU, log-log)",
        marker = :circle,
        legend = :topleft,
        linewidth = 2,
        yscale = :log10
    )

    ops_path = joinpath(output_dir, "lu_factorization_operations_vs_size.png")
    ops_log_path = joinpath(output_dir, "lu_factorization_operations_vs_size_log.png")
    time_path = joinpath(output_dir, "lu_factorization_time_vs_size.png")
    time_log_path = joinpath(output_dir, "lu_factorization_time_vs_size_log.png")
    savefig(p_ops, ops_path)
    savefig(p_ops_log, ops_log_path)
    savefig(p_time, time_path)
    savefig(p_time_log, time_log_path)
    display(p_ops)
    display(p_ops_log)
    display(p_time)
    display(p_time_log)
    return (operations_plot = ops_path, operations_plot_log = ops_log_path, time_plot = time_path, time_plot_log = time_log_path)
end

function run_experiment(; max_size::Int = CONFIG.max_size, step::Int = CONFIG.step, trials::Int = CONFIG.trials, random_seed::Int = CONFIG.random_seed, output_dir::AbstractString = CONFIG.output_dir)
    @assert max_size ≥ 1 "max_size must be ≥ 1"
    @assert step ≥ 1 "step must be ≥ 1"
    @assert trials ≥ 1 "trials must be ≥ 1"
    Random.seed!(random_seed)

    sizes = build_sizes(max_size, step)
    records = NamedTuple{(:size,:trial,:adds,:muls,:time),Tuple{Int,Int,Int,Int,Float64}}[]
    for n in sizes
        for trial in 1:trials
            A = generate_matrix(n)
            add_ref = Ref(0)
            mul_ref = Ref(0)
            elapsed = @elapsed begin
                _, _, _, _ = LU_factor(A, default_mul, add_ref, mul_ref)
            end
            push!(records, (size = n, trial = trial, adds = add_ref[], muls = mul_ref[], time = elapsed))
        end
    end

    summary = summarize(records, sizes)
    outputs = persist_results(records, summary; output_dir = output_dir, max_size = max_size, step = step, trials = trials, random_seed = random_seed)
    plot_paths = plot_results(summary, sizes; output_dir = output_dir)
    return (records = records, summary = summary, files = outputs, plots = plot_paths)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment()
end