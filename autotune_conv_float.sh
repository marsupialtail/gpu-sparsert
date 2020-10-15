#!/bin/bash
IC=$1
OC=$2
IMAGE_DIM=$3
fil=$4

#for PACT submission no fusion of bias and relu
python3 scripts/make_conv_input.py $IC $OC $IMAGE_DIM rn90_3x3/95/$fil

RESIDUAL=0

if [ $IMAGE_DIM -eq 56 ] ; then
#        C_choices=(28 49 98)
	C_choices=(56)
fi
if [ $IMAGE_DIM -eq 28 ] ; then
#        C_choices=(2 7 14)
	C_choices=(28)
fi
if [ $IMAGE_DIM -eq 14 ] ; then
#        C_choices=(2 7 14)
	C_choices=(14)
fi
if [ $IMAGE_DIM -eq 7 ] ; then
#        C_choices=(2 7 14)
	C_choices=(1)
fi

#A_choices=($(($OC / 32)) $(($OC / 16)) $(($OC / 8)))
A_choices=($(($OC / 64)) $(($OC / 32)) $(($OC / 16)))
Gy=1
for A in ${A_choices[@]}; do
	for C in ${C_choices[@]}; do
	
		kernel_name=$(echo $fil | cut -d "." -f1)_${A}_${C}.cubin
		echo $kernel_name
		ptx_name=$(echo $fil | cut -d "." -f1)_${A}_${C}.ptx
		echo $ptx_name

		echo $Gy $A $C

		python3 sparsednn/code_gen_conv_ptx.py --Gy=$Gy --IC=$IC --OC=$OC --IMAGE_DIM=$IMAGE_DIM --infile transposed_filter.npy --outfile $ptx_name --A_blocks=$A --C_blocks=$C 
		ptxas -arch=sm_75 $ptx_name -o ${kernel_name}
		cp ${kernel_name} testing_conv.cubin

		ADDITIONAL_INCLUDES=""

		if ! [ -z ${CUDNN_INCDIR+x} ];
		then ADDITIONAL_INCLUDES="${ADDITIONAL_INCLUDES} -I ${CUDNN_INCDIR}";
		fi

		if ! [ -z ${CUDNN_LIBDIR+x} ];
		then ADDITIONAL_INCLUDES="${ADDITIONAL_INCLUDES} -L ${CUDNN_LIBDIR}";
		fi

		nvcc -std=c++11 -I build/include -L build/lib -O3 -arch=sm_75 sparsednn/driver_conv.cu -lcudnn -lcnpy -lcuda -DRESIDUAL=$RESIDUAL,HALF=0,A_Blocks=$A,C_Blocks=$C,Gy=$Gy,IC=$IC,OC=$OC,IMAGE_DIM=${IMAGE_DIM} -o ./exe ${ADDITIONAL_INCLUDES}
		LD_LIBRARY_PATH=$LD_LIBRARY_PATH:build/lib ./exe
		python3 scripts/test_equivalence.py kernel_output.npy cudnn_output.npy
	done
done


