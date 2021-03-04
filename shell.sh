
for width in $(seq 5 1 260);
do
	./winograd_dev2 32 120 $width 1 
done
