`timescale 1ns/1ps

module LIF_Neuron_tb;

    // Testbench signals
    reg clk;
    reg reset;
    reg [7:0] input_current;
    wire spike;

    // Instantiate the DUT (Device Under Test)
    LIF_Neuron dut (
        .clk(clk),
        .reset(reset),
        .input_current(input_current),
        .spike(spike)
    );

    // Clock generation
    always #5 clk = ~clk; // 10 ns clock period

    // Testbench sequence
    initial begin
        // Enable waveform generation
        $dumpfile("waveform.vcd");
        $dumpvars(0, LIF_Neuron_tb);

        // Initialize signals
        clk = 0;
        reset = 1;
        input_current = 0;

        // Apply reset
        #10 reset = 0;

        // Test case 1: Low input, no spike
        input_current = 8'd10;
        #100;

        // Test case 2: High input, trigger spike
        input_current = 8'd50;
        #100;

        // Test case 3: Reset during operation
        reset = 1;
        #10 reset = 0;
        #100;

        // Test case 4: Alternating inputs
        input_current = 8'd30;
        #50;
        input_current = 8'd0;
        #50;

        // End simulation
        $finish;
    end

    // Monitor signals
    initial begin
        $monitor($time, " Reset=%b, Input=%d, Membrane Potential=%d, Spike=%b",
                 reset, input_current, dut.membrane_potential, spike);
    end

endmodule
