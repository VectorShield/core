for i in $(find . | grep -E "(web_ui.py|.css|.html)");
do
    echo $i
    cat $i
    echo
    echo
done > all-code.txt
