#!/bin/bash

# Navigate to src/scripts directory
cd ./src/scripts/

NSTARS=2

NAME="yZ1-c11-fiducial"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --agb-yields=cristallo11
echo ""

NAME="yZ1-c11-lowsfe"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --agb-yields=cristallo11 --sfe-factor=2
echo ""

NAME="yZ1-c11-mscale"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --agb-yields=cristallo11 --agb-mscale=0.5
echo ""

NAME="yZ1-c11-Zscale"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --agb-yields=cristallo11 --agb-Zscale=2
echo ""

NAME="yZ1-c11-onlyagb"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --agb-yields=cristallo11 --r-channel=none
echo ""

NAME="yZ1-c11-rdelay-plaw"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --agb-yields=cristallo11 --r-channel=sneia --RIa=powerlaw --minimum-delay=0.01
echo ""

NAME="yZ1-c11-lateburst"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ1 --agb-yields=cristallo11 --evolution=lateburst
echo ""

NAME="yZ2-c11-x2"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --agb-yields=cristallo11 --agb-amp=2
echo ""

NAME="yZ2-k16-fiducial"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --agb-yields=karakas16
echo ""

NAME="yZ2-k16-lowsfe"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --agb-yields=karakas16 --sfe-factor=2
echo ""

NAME="yZ2-k16-mscale"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --agb-yields=karakas16 --agb-mscale=0.5
echo ""

NAME="yZ2-k16-Zscale"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --agb-yields=karakas16 --agb-Zscale=2
echo ""

NAME="yZ2-k16-onlyagb"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --agb-yields=karakas16 --r-channel=none
echo ""

NAME="yZ2-k16-rdelay-plaw"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --agb-yields=karakas16 --r-channel=sneia --RIa=powerlaw --minimum-delay=0.01
echo ""

NAME="yZ2-k16-lateburst"
echo $NAME
python -m multizone -f --nstars=$NSTARS --name=$NAME --yields=yZ2 --agb-yields=karakas16 --evolution=lateburst
echo ""
