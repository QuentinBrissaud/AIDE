# Artificial intelligence for Ionospheric perturbations DEtections (AIDE)

## Summary
AIDE allows you to scan vTEC records to detect co-seismic ionospheric perturbations. AIDE software consists of a Machine-Learning based detector, arrival-time picker, and an associator across GNSS satellite networks. 

<p align="center">
<img width="421" alt="image" src="https://user-images.githubusercontent.com/6717390/230471273-a553ba77-6860-4752-8100-3eb6956b4a40.png">
</p>
Figure: (from Brissaud and Astafyeva, 2022) (d) hand-picked arrival times for satellites G05 and G26 along with the epicentre location
(yellow star), and surface projection of the fault slip (in m) as green to yellow patches, (e) RF-based arrival-time predictions for each confirmed detection
for satellites G05, G26 and G27 with an inset plot showing a newly detected CID arrival (red vertical line) for satellite G27 and station 0167 which was not
reported by human analyst and (f) association classes determined from confirmed detections, along with an inset plot showing the vTEC data for satellite G26,
station 0155.

## Requirements
- Python3.7
- pandas
- obspy
- sklearn
- multiprocessing
- seaborn

## Usage
See Python notebook "AIDE_nb.ipynb"

## Data
You can collect the vTEC data and random forest models here: https://doi.org/10.6084/m9.figshare.19661115

## Paper
Tsunamis generated by large earthquake-induced displacements of the ocean floor can lead to tragic consequences for coastal communities. Ionospheric measurements of Co-Seismic Disturbances (CIDs) offer a unique solution to characterize an earthquake’s tsunami potential in Near-Real-Time (NRT) since CIDs can be detected within 15 min of a seismic event. However, the detection of CIDs relies on human experts, which currently prevents the deployment of ionospheric methods in NRT. To address this critical lack of automatic procedure, we designed a machine-learning based framework to (1) classify ionospheric waveforms into CIDs and noise, (2) pick CID arrival times, and (3) associate arrivals across a satellite network in NRT. Machine-learning models (random forests) trained over an extensive ionospheric waveform dataset show excellent classification and arrival-time picking performances compared to existing detection procedures, which paves the way for the NRT imaging of surface displacements from the ionosphere.
https://doi.org/10.1093/gji/ggac167

## Citation
Brissaud, Q., & Astafyeva, E. (2022). Near-real-time detection of co-seismic ionospheric disturbances using machine learning. Geophysical Journal International, 230(3), 2117-2130. doi: 10.1093/gji/ggac167
```
@article{brissaud2022near,
  title={Near-real-time detection of co-seismic ionospheric disturbances using machine learning},
  author={Brissaud, Quentin and Astafyeva, Elvira},
  journal={Geophysical Journal International},
  volume={230},
  number={3},
  pages={2117--2130},
  year={2022},
  publisher={Oxford University Press}
}
```
