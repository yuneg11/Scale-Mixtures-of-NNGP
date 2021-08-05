data=$1
gpu=$2
seed=10

for nh in 1 4 16; do
    for wv in 1 2 4; do
        for bv in 0. 0.09 1.0; do
            for lv in 1. 4.; do
                for act in "erf" "relu"; do
                    for e in 5 6 7 8 9; do
                        python eval.py reg \
                                $data \
                                -nh $nh \
                                -wv $wv \
                                -bv $bv \
                                -a "[0.5, 1., 1.5, 2., 4., 8.]" \
                                -b "[0.5, 1., 1.5, 2., 4., 8.]" \
                                -bc "[1., 2., 3.]" \
                                -bd "[1., 2., 3.]" \
                                -lv $lv \
                                -act $act \
                                -e $e \
                                -g $gpu \
                                -f 1.0 \
                                -s $seed
                    done
                done
            done
        done
    done
done
