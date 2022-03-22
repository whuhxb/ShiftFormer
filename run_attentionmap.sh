for img in demo1.JPEG demo2.JPEG demo3.JPEG demo4.JPEG demo5.JPEG demo6.JPEG
do
    for Q in 23 83 143
    do
        for L in 0 6 11
        do
            for H in 0 3 6 9
            do
                python3 single.py --image $img --query $Q --layer $L --head $H
            done
        done
    done
done