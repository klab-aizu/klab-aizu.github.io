#include "platform.h"
#include "xgpio.h"
#include "xparameters.h"

// Instantiation
XGpio LEDInst;

int main () {
	init_platform();

	int status;
	// Initialization
	status = XGpio_Initialize(&LEDInst, XPAR_AXI_GPIO_0_DEVICE_ID);

	if (status != XST_SUCCESS) {
		return XST_FAILURE;
	}

	// Set Ch.1 as Output
	XGpio_SetDataDirection(&LEDInst, 1, 0);

	int bit = 0x1;
	int loop = 10;

	print("The program for LEDs is running...\n\n");
	while (loop--) {
		for (int i = 0; i < 16; i++) {
			XGpio_DiscreteWrite(&LEDInst, 1, (u32)bit);  // On
			for (int j = 0; j < 2500000; j++) {} // Latency
			XGpio_DiscreteWrite(&LEDInst, 1, 0);  // Off
			if (bit == 0x8000)
				bit = 0x1;
			else
				bit = bit << 1;
		}
	}
	print("The program was completed successfully.\n");

	cleanup_platform();
	return 0;
}
