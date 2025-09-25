#!/bin/bash

# ============================================================================
# Setup
#
# SRC_DIR : The directory to be backed up. Soft links are followed, so the
#           files they point to are included in the backup.
# DST_DIR : Where to put the tar archive and associated CONTENTS file.
# ============================================================================

# SET DEFAULTS:
SRC_DIR=/GLM
DST_DIR=/atap-backup
TMP_DIR=/tmp

# MISC. INITIALIZATION:
PROGNAME=`basename $0`
DATE=`date +"%Y-%m-%d-%H%M%S"`
HELP="false"
REF_TIME=null

# PARSE COMMAND LINE OPTIONS:
set -- `getopt "hs:d:r:" "$@"`
while :
do
        case "$1" in
        -h | --help)    HELP="true" ;;  # Print usage.
        -s | --source)  shift; SRC_DIR="$1" ;;  # Source directory.
        -d | --dest)    shift; DST_DIR="$1" ;;  # Destination directory.
        -r | --reftime) shift; REF_TIME="$1" ;; # Optional time of last backup "%Y-%m-%d %H:%M:%S"
        --) break ;;
        esac
        shift
done
shift # REMOVE THE TRAILING --

# PRINT HELP:
if [ $HELP = "true" ]
then
    echo "Usage: $PROGNAME" \
	 "[-h] [-s source_directory] [-d destination_directory] [-r reference_tim]" 1>&2
	echo -e '\t -r "%Y-%m-%d %H:%M:%S" (example: "2000-01-01 00:00:00")'
    exit 1
fi

# SPECIFY REFERENCE FILE. INITIALIZE REF TIME, IF NECESSARY:
REF_FILE=${SRC_DIR}/.last-backup-time
echo "Reference File: $REF_FILE"

if [ $REF_TIME = "null" ]
then
    if [ -s "$REF_FILE" ]  # Returns 0 if file exists and is not empty.
    then
    	REF_TIME=`cat $REF_FILE`
    else
        REF_TIME="2000-01-01 00:00:00"
    fi
fi

# SPECIFY TEMP FILES:
TMP_LIST="${TMP_DIR}/${PROGNAME}.$$.tmp"
TMP_ARCHIVE="${TMP_DIR}/atap-backup-${DATE}.tar.gz"
TMP_CONTENTS="${TMP_DIR}/CONTENTS-${DATE}.txt"

# ============================================================================
# Make a list of files to archive.
#
# * The -L option causes find to evaluate symlink file types as those of the 
#   file referenced by the link, not the link itself. The '-type f' option 
#   will therefore include symlink files in the output.
# ============================================================================
echo -e "find -L $SRC_DIR -newermt \"$REF_TIME\" -type f -print >  $TMP_LIST"
find -L $SRC_DIR -newermt "$REF_TIME" -type f -print >  $TMP_LIST

# ============================================================================
# Create the archive or do nothing if no modified files were detected.
# ============================================================================
if [ -s "$TMP_LIST" ] # Returns 0 if file exists and is not empty.
then
    echo "tar -czvhf $TMP_ARCHIVE -T $TMP_LIST --mode='a+rwX' | tee $TMP_CONTENTS"
	tar -czvhf $TMP_ARCHIVE -T $TMP_LIST --mode='a+rwX' | tee $TMP_CONTENTS

	# Copy the archive to lou
	echo "scp $TMP_ARCHIVE $TMP_CONTENTS $DST_DIR"
	scp $TMP_ARCHIVE $TMP_CONTENTS $DST_DIR
	
	rm $TMP_ARCHIVE $TMP_CONTENTS
else
	echo "Nothing to back up as of `date`"
fi

# Update the reference time.
date +"%Y-%m-%d %H:%M:%S" > $REF_FILE 

rm $TMP_LIST

# EOF
