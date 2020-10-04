#!/bin/bash

infile=mobilenet/contraction_1x1_$1_transposed.npy

if [ $1 -eq 0 ]; then
	A_dim=64
	B_dim=32
	C_dim=12544
fi
if [ $1 -eq 1 ]; then
	A_dim=128
	B_dim=64
	C_dim=3136
fi
if [ $1 -eq 2 ]; then
	A_dim=128
	B_dim=128
	C_dim=3136
fi
if [ $1 -eq 3 ]; then
	A_dim=256
	B_dim=128
	C_dim=784
fi
if [ $1 -eq 4 ]; then
	A_dim=256
	B_dim=256
	C_dim=784
fi
if [ $1 -eq 5 ]; then
	A_dim=512
	B_dim=256
	C_dim=196
fi
if [ $1 -eq 6 ]; then
	A_dim=512
	B_dim=512
	C_dim=196
fi
if [ $1 -eq 11 ]; then
	A_dim=1024
	B_dim=512
	C_dim=49
fi
if [ $1 -eq 12 ]; then
	A_dim=1024
	B_dim=1024
	C_dim=49
fi

if [ $C_dim -eq 12544 ] ; then
        C_choices=(49 56 98)
fi
if [ $C_dim -eq 3136 ] ; then
        C_choices=(14 28 56)
fi
if [ $C_dim -eq 784 ] ; then
        C_choices=(2 7 14)
fi
if [ $C_dim -eq 196 ] ; then
        C_choices=(1 2)
fi
if [ $C_dim -eq 49 ] ; then
        C_choices=1
fi
if [ $C_dim -eq 56 ] ; then
        C_choices=1
fi


python scripts/make_BC.py $infile $C_dim

A_choices=($(($A_dim / 32)) $(($A_dim / 16)) $(($A_dim / 8)))
besttime=100
#for A_blocks in ${A_choices[@]}; do

#	for C_blocks in ${C_choices[@]}; do
for A_blocks in $(($A_dim / 8)); do

	for C_blocks in $((C_dim / 49)); do
		for Gy in 1; do

			#./test > runtime
			python sparsednn/code_gen_ptx.py --A_dim $A_dim --B_dim $B_dim --C_dim $C_dim --A_blocks $A_blocks --C_blocks $C_blocks --Gy $Gy --infile $infile --outfile testing.ptx
			ptxas -arch=sm_75 testing.ptx -o testing.cubin
                        nvcc sparsednn/driver_spmm.cpp -w -O3 -DA_dim=$A_dim,B_dim=$B_dim,C_dim=$C_dim,A_Blocks=$A_blocks,C_Blocks=$C_blocks,Gy=$Gy,infile=$infile_str -lcuda -lcudart -lcnpy -o exe --std=c++11 -Xptxas="-v" -I ../cnpy -L ../cnpy/build
			./exe > runtime
			cat runtime
			python scripts/test_equivalence.py ref.npy ptx_result.npy
			runtime=$(grep "kernel used" runtime | awk '{print $3}')
			echo $runtime
			if (( $(echo "$runtime < $besttime" | bc -l) )) ; then 
				besttime=$runtime
			fi
		done
	done
done

echo Best Runtime $besttime
