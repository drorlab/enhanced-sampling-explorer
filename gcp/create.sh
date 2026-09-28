#!/usr/bin/env bash
# Create the CPU spot VM for the demo. Tries zones in order until one has capacity.
#   bash gcp/create.sh
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${NAME:-esdemo}
ZONES=${ZONES:-us-central1-a,us-central1-b,us-central1-c,us-central1-f}
MACHINE=${MACHINE:-c3-highcpu-88}
for ZONE in ${ZONES//,/ }; do
  echo "=== trying $MACHINE in $ZONE ==="
  if gcloud compute instances create "$NAME" --zone="$ZONE" --machine-type="$MACHINE" \
      --image-family=debian-12 --image-project=debian-cloud \
      --boot-disk-size=100GB --boot-disk-type=pd-balanced \
      --provisioning-model=SPOT --instance-termination-action=STOP \
      --scopes=cloud-platform --labels=purpose=esdemo \
      --metadata-from-file=startup-script=gcp/startup.sh,env-yml=environment.yml; then
    echo "$ZONE" > gcp/.zone
    echo "created $NAME in $ZONE; follow: gcloud compute ssh $NAME --zone $ZONE -- sudo tail -f /var/log/esdemo-startup.log"
    exit 0
  fi
done
echo "no capacity in any of $ZONES" >&2; exit 1
