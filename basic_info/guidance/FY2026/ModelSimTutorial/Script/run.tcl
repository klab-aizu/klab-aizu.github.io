vlib work
vmap work work
vlog Circuit/*.v
vlog Testbench/*.v 
vsim -voptargs="+acc" work.tb_top
add wave -group {tb_top} tb_top/*
add wave -group {dut} tb_top/dut/*
add wave -group {u_and} tb_top/dut/u_and/*
add wave -group {u_or} tb_top/dut/u_or/*
run -all