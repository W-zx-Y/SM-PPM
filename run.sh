#!/bin/sh

declare -i iter=500

for j in {1..5}
do
    for i in {1..5}
    do
        python train.py --save-pred-every $iter --num-steps-stop $iter --snapshot-dir './snapshots/exp'$i'_shot'$j --data-list-target './dataset/cityscapes_list/trainOne'$j'.txt'

        python evaluate_cityscapes.py --save './result/' --restore-from './snapshots/exp'$i'_shot'$j'/GTA5_'$iter'.pth' --log 'log.txt'
    done
done

