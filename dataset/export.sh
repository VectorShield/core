for i in $(find . | grep -E "(.py$)");
do
    echo $i
    cat $i
    echo
    echo
done > all-code.txt
