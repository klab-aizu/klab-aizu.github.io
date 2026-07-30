module Izhikevich_Neuron (
    input clk,
    input reset,
    input signed [15:0] input_current,
    output reg signed [15:0] v, // Membrane potential
    output reg spike
);
    // Parameters
    parameter signed [15:0] a = 16'sd2;   // Recovery time scale
    parameter signed [15:0] b = 16'sd2;   // Sensitivity of u
    parameter signed [15:0] c = -16'sd65; // Reset value for v
    parameter signed [15:0] d = 16'sd8;   // Reset increment for u
    parameter signed [15:0] v_threshold = 16'sd30;

    // Internal variables
    reg signed [15:0] u; // Recovery variable

    // Update logic
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            v <= -16'sd70; // Initial membrane potential
            u <= 16'sd0;   // Initial recovery variable
            spike <= 1'b0;
        end else begin
            // Spike condition
            if (v >= v_threshold) begin
                v <= c;          // Reset membrane potential
                u <= u + d;      // Increment recovery variable
                spike <= 1'b1;   // Output spike
            end else begin
                spike <= 1'b0;
                // Update equations
                v <= v + ((16'sd4 * v * v) / 16'sd10) + (16'sd5 * v) + 16'sd140 - u + input_current;
                u <= u + a * (b * v - u);
            end
        end
    end
endmodule
