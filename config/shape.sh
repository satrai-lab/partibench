#!/bin/bash

# TO RUN:
# chmod +x shape_multi.sh
# ./shape_multi.sh rules.tsv

FILE=$1
IF="ens5"

create() {

	tc qdisc add dev $IF root handle 1: htb default 10  # root qdisc

	line_number=-1  # to count node links and also skip header line

	while IFS=$'\t' read -r DEST_NAME IP RATE DELAY; do  # read TSV values from file

		echo "DEST_NAME: $DEST_NAME, IP: $IP, RATE: $RATE"

		((line_number++))

		# Skip header line
		if [ $line_number -eq 0 ]; then
			continue
		fi

		echo "=== INIT SHAPING FOR LINK TO $DEST_NAME (#$line_number) ==="

		# Apply outbound link shaping rules for current destination node
		tc class add dev $IF parent 1: classid 1:$line_number htb rate $RATE
		#tc qdisc add dev $IF parent 1:$line_number handle $((line_number * 10)): netem delay $DELAY
		tc filter add dev $IF protocol ip parent 1:0 prio 1 u32 match ip dst $IP flowid 1:$line_number

		echo "=== SHAPING FOR LINK TO $DEST_NAME (#$line_number) SUCCESSFULL!==="

	done < "$FILE"
}

clean() {
	echo '== CLEANING =='

	tc qdisc del dev $IF root  # remove everything by removing root qdisc
}

clean
create
