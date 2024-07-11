#### Run Localization for Untrained Models
for level in query key value attn
do 
    python3 responses.py arch=bert-c5_topo +stimuli=moseley +level=$level impl.sharing_strategy=file_system
done
