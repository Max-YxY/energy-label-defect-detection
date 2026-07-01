#!/bin/bash

stedgeai generate --model best13.onnx --target stm32n6 --st-neural-art default@user_neuralart.json --input-data-type uint8 --output-data-type float32 --inputs-ch-position chlast
cp st_ai_output/network_ecblobs.h .
cp st_ai_output/network.c .
cp st_ai_output/network_atonbuf.xSPI2.raw network_data.xSPI2.bin
arm-none-eabi-objcopy -I binary network_data.xSPI2.bin --change-addresses 0x70200000 -O ihex ../Binary/network-data.hex