for img in images/demo1.JPEG images/demo2.JPEG images/demo3.JPEG images/demo4.JPEG images/demo5.JPEG images/demo6.JPEG
do
#    for Q in 23 83 143
    for Q in 98
    do
        for L in 0 6 11
        do
            for H in 0 3 6 9
            do
#                echo $img $Q $L $H
               python3 single.py --image $img --query $Q --layer $L --head $H
            done
        done
    done
done
