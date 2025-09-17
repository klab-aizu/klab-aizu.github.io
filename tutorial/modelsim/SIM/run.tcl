vlib work
vmap work work
vlog ../SRC/Flip_Flop.v  
vlog ../TB/TestBench.v
vsim -voptargs="+acc" work.TestBench
add wave /*
run -all
