# #!/bin/sh

# #source env/bin/activate

# for line in $(ls -Art profiles | tail --n $1)
# do
# 	python img_from_txt.py -p profiles/$line
# done

#!/bin/sh

# Create the images_experiment directory if it does not exist
mkdir -p images_experiment

# Iterate over the files in the profiles directory
#!/bin/bash

# Ensure you are looking in the specific directory
for line in $(ls -Art profiles/impulses_volt | tail -n $1)
do
    # Skip if it's a directory
    if [ -d "profiles/impulses_volt/$line" ]; then
        echo "Skipping directory: profiles/impulses_volt/$line"
        continue
    fi
    
    # Run the Python script
    python img_from_txt.py -p profiles/impulses_volt/$line
done

