#!/bin/bash
#
# Script to submit a job for every single experiment that needs to be done.
#
# David Duvenaud
# March 2011
# ==================================

. /usr/local/grid/divf2/common/settings.sh 

for r in 1
do
	# Classification jobs
	for p in {1..16}
	do
		for m in {1..7}
		do
		    qsub -l lr=0 -o "logs/run_log_$p_$r_$r.txt" -e "logs/error_log_$p_$r_$r.txt" run_one_job.sh $p $m 100 $r
		done
	done
done

