#!/bin/bash

tmpFile=/tmp/hosts_`whoami`
tmpFile2=/tmp/hosts2_`whoami`

# Discover all the hosts matching the supplied IP pattern(s)
nmap -oG ${tmpFile} $* >/dev/null

# Get the IP of the localhost. It has to participate in the launch 
ifconfig | head -n 2 | tail -n 1 | gawk -F "addr:" '{print $2}' | gawk '{print $1}' > ${tmpFile2}

# Filter out the hosts not supporting SSH
grep ssh ${tmpFile} | gawk -F " " '{print $2}' >> ${tmpFile2}

# Remove -if it exists- a duplicate entry for localhost
uniq ${tmpFile2} ${tmpFile}

# Get the cores for each host in the temporary file
for h in `cat ${tmpFile}`
do
    res=`ssh -o ConnectTimeout=5  -o BatchMode=yes -o StrictHostKeyChecking=no $h cat /proc/cpuinfo | grep processor | wc | gawk '{print $1}'`

    # Output the IP only if there is a valid response from the previous command
    if [ "${res:-0}" -ne 0 ]
       then 
       echo "$h slots= ${res}"
    fi
done
