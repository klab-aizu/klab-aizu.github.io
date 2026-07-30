#!/bin/bash

# Set the file names
VERILOG_FILES="Izhikevich_Neuron.v Izhikevich_Neuron_tb.v"
OUTPUT_EXEC="simulation.out"
VCD_FILE="waveform.vcd"

# Step 1: Compile the Verilog files
iverilog -o $OUTPUT_EXEC $VERILOG_FILES

# Step 2: Run the simulation
vvp $OUTPUT_EXEC

# Step 3: View the waveform (optional)
if [ -f $VCD_FILE ]; then
    echo "Waveform file generated: $VCD_FILE"
    echo "Use a waveform viewer like GTKWave to analyze the simulation."
    echo "To view the waveform, run: gtkwave $VCD_FILE"
else
    echo "No waveform file generated. Check the testbench for VCD generation commands."
fi

