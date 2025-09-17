`timescale 1ns/1ps

module Izhikevich_Neuron_tb;

    // Testbench signals
    reg clk;
    reg reset;
    reg signed [15:0] input_current;
    wire signed [15:0] v;
    wire spike;

    // Instantiate the DUT (Device Under Test)
    Izhikevich_Neuron dut (
        .clk(clk),
        .reset(reset),
        .input_current(input_current),
        .v(v),
        .spike(spike)
    );

    // Clock generation
    always #5 clk = ~clk; // 10 ns clock period

    // Testbench sequence
    initial begin
        // Enable waveform generation
        $dumpfile("waveform.vcd");
        $dumpvars(0, Izhikevich_Neuron_tb);

        // Initialize signals
        clk = 0;
        reset = 1;
        input_current = 0;

        // Apply reset
        #10 reset = 0;

        // Test case 1: Small input current
        input_current = 16'sd5;
        #100;

        // Test case 2: Larger input current
        input_current = 16'sd20;
        #100;

        // Test case 3: Reset during operation
        reset = 1;
        #10 reset = 0;
        #100;

        // End simulation
        $finish;
    end

    // Monitor signals
    initial begin
        $monitor($time, " Reset=%b, Input=%d, v=%d, Spike=%b",
                 reset, input_current, v, spike);
    end

endmodule
