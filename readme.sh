rm -rf name.txt
rm -rf README.md
touch name.txt
touch README.md

find ./ -name "*.py" -print | more >> name.txt
for name in `cat name.txt`
do
    echo "# $name" >> README.md
    echo "------" >> README.md
    echo "\`\`\`python" >> README.md
    cat $name >> README.md
    echo "\`\`\`" >> README.md
done
rm -rf name.txt
