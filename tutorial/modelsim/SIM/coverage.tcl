vlib work 
vmap work work
vlog +cover ../SRC/Flip_Flop.v 
vlog ../TB/TestBench.v
vsim -coverage -voptargs="+acc" work.TestBench
add wave /*
run -all
coverage report -details -output coverage_report.txt
