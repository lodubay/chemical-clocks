#!/bin/bash

# Navigate to src/scripts directory
cd ./src/scripts/

NSTARS=2
EVOLUTION=insideout

NAME="fiducial"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=fiducial --evolution=$EVOLUTION
echo ""

NAME="low-sfe"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --sfe-factor=2
echo ""

NAME="agb-mscale"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=mscale
echo ""

NAME="agb-Zscale"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=Zscale
echo ""

NAME="agb-only"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=onlyagb
echo ""

NAME="lateburst"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --evolution=lateburst
echo ""
