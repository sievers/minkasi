### Any user should be able to enter the following commands (within a terminal)
### And get results.

##############     ACTIVATE THE RELEVANT ENVIRONMENT    ################
$ cd /users/cromero/GitHubRepos/minkasi  # Not strictly necessary
# (If you wish to give full paths when calling the scripts/yaml files)

$ source /opt/local/stow/anaconda3/etc/profile.d/conda.sh
$ conda activate /users/cromero/.conda/envs/minkasi_fftw

### Accounts will often set their own pythonpaths (via .bash_profile)
### which may conflict with the desired path. (NB, it seems better to
### do this step after sourcing conda.sh and activating the conda environment.)
###############       RESET PYTHONPATH          ########################
$ unset PYTHONPATH
$ export PYTHONPATH="/users/cromero/GitHubRepos/minkasi:/users/cromero/.conda/envs/minkasi_fftw/lib/python311.zip:/users/cromero/.conda/envs/minkasi_fftw/lib/python3.11:/users/cromero/.conda/envs/minkasi_fftw/lib/python3.11/lib-dynload:/users/cromero/.conda/envs/minkasi_fftw/lib/python3.11/site-packages"

###############              RUN THE SCRIPTS         ####################
$ python examples/fit_gaussians_with_yaml.py examples/macs0647_gaussians.yaml
$ python examples/fit_rings_with_yaml.py examples/macs0647_rings.yaml

##########################################################################
#######   (ALTERNATIVE) RUN THE SCRIPTS **WITH** PROFILER    #############
#######   OPTIONALLY: (If you want to profile the results):   ############
$ date_string=$(date +"%d%b%Y")
$ python -m cProfile -s cumulative -o "myprofile1_$date_string.dat" examples/fit_gaussians_with_yaml.py examples/macs0647_gaussians.yaml
$ python -m cProfile -s cumulative -o "myprofile2_$date_string.dat" examples/fit_rings_with_yaml.py examples/macs0647_rings.yaml

### I leave it to your own knowledge or Googling/LLM usage to find how to
### read the profile results and how one make adjustments. This avenue may
### lead to desires to change code, which is more than intended for the
### average user.

##########################################################################
########            ADAPTING CODE FOR OTHER DATASETS           ###########
##########################################################################

##########                 NOTA BENE                           ###########
# An overarching concern is simply making sure you have the correct file
# names and paths (to the yaml files and within the yaml files).

########            SAME DATASET, DIFFERENT OPTIONS            ########### 
# If you just want to tinker with options, you can make a copy of each yaml
# file (in your own directory), edit those (yaml) files,
# and use the full path(s) in the python launch command.
# --- FOR EXAMPLE
# If I logged in as penarray, I could
$ cp /users/cromero/GitHubRepos/minkasi/examples/macs0647_gaussians.yaml /users/penarray/Python/scripts_yamls/macs0647_gaussians.yaml
# Edit that yaml file with your favorite IDE/text editor.
# and then run:
$ python examples/fit_gaussians_with_yaml.py /users/penarray/Python/scripts_yamls/macs0647_gaussians.yaml

########                   DIFFERENT DATASET(S)                ###########
# The major point about a different dataset is that, very likely, you will
# need to produce the (calibrated) TODs for this. The team will take care
# of creating calibration files. However, you will still need to run the
# processing (which incorporates the calibration files). In the simplest
# form it will be something like:

$ /users/penarray/Public/m2/reduction_scripts/shell_integration/find_src -M MACS0647*

# replacing MACS0647* with your target name (followed by the asterisk).
# This will generate the files, and it should be in a directory like
# /home/scratch/<USER>/mustang/MUSTANG2/Reductions/<TARGET>/TS_EaCMS0f0_51_<DATE>/
# You'll need to update the yaml files accordingly.

### NOTA BENE:
# find_src will, by default, output to your current (present) working directory. Moreover, it generally
# wants that directory to not have previous files created by find_src.
# Two points:
# ONE: I would make sure you're somewhere in your scratch directory (/home/scratch/<USER>/)
# TWO: I like to make a new directory before I run find_src.
# In particular, I (cromero) tend to put stuff in:
# /home/scratch/cromero/mustang/MUSTANG2/FindSrcOutputs/
# And within that directory, I might
$ mkdir TestDir
# so that now the outputs get written to /home/scratch/cromero/mustang/MUSTANG2/FindSrcOutputs/TestDir/

# For more information, see:
# https://gbtdocs.readthedocs.io/en/latest/how-tos/receivers/mustang2/data/mustang2_find_src.html

########                MANUALLY CHECKING TODS                  ###########
# While it can be a lot of work, it's worthwhile inspecting each scan to
# see if there are scans that should be omitted.

# If you run find_src with the MIDAS mapmaker, e.g.:
$ /users/penarray/Public/m2/reduction_scripts/shell_integration/find_src -m MACS0647*
# , it'll produce a text output file of the (MIDAS! not Minkasi!) map noise.
# Although there are meaningful differences between MIDAS and Minkasi, the MIDAS map noise can give
# some (wavy/shaky hand gesture) indication of the data quality of that scan. For example, you can
# have a look at this file:
/home/scratch/cromero/mustang/MUSTANG2/FindSrcOutputs/M0647_redo/NoisePerScan_MIDAS_0f070-to-49f0Hz.txt
# It takes a while to get a sense for "good" numbers. The easiest place to start (especially if there are
# more than ~10 scans) is simply to let your eye catch the outliers (especially towards higher noise values
# *or* really low numbers of detectors (compared to other scans in the list).

# Running this (make sure you have a clean directory to run from):
$ /users/penarray/Public/m2/reduction_scripts/shell_integration/find_src -m -C '/users/penarray/Public/m2/reduction_scripts/shell_integration/MIDAS_TOD_plot_conf.pro' MACS0647*
# invokes slightly different parameters that will produce and save PNG files of the TODs (per ROACH).
