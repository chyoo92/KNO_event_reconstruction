import os
import ROOT
import numpy as np
import h5py
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, required=True, help='Path to input directory')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output directory')
args = parser.parse_args()



ROOT.gSystem.Load("/users/hep/yewzzang/WCSim_KNO/WCSim_v1.8.0-build/libWCSimRoot.so")

fin = ROOT.TFile(args.input)

eventT = fin.Get("wcsimT")
event = ROOT.WCSimRootEvent()
eventT.SetBranchAddress("wcsimrootevent", event)
eventT.GetBranch("wcsimrootevent").SetAutoDelete(1)
eventT.GetEntry(0)
nEvents = eventT.GetEntries()

## Load the geometry
geomT = fin.Get("wcsimGeoT")
geom = ROOT.WCSimRootGeom()
geomT.SetBranchAddress("wcsimrootgeom", geom)
geomT.GetEntry(0)
nODPMTs = geom.GetODWCNumPMT()
nPMTs = geom.GetWCNumPMT()

print("--------------------")
print(f" nEvents = {nEvents}")
print(f" nPMTs   = {nPMTs}")
print(f" nODPMTs = {nODPMTs}")
print("--------------------")

out_pmt_x = np.zeros(nPMTs)
out_pmt_y = np.zeros(nPMTs)
out_pmt_z = np.zeros(nPMTs)

out_pmt_px = np.zeros(nPMTs)
out_pmt_py = np.zeros(nPMTs)
out_pmt_pz = np.zeros(nPMTs)

for iPMT in range(nPMTs):
    pmt = geom.GetPMT(iPMT)
    out_pmt_x[iPMT] = pmt.GetPosition(0)
    out_pmt_y[iPMT] = pmt.GetPosition(1)
    out_pmt_z[iPMT] = pmt.GetPosition(2)
    out_pmt_px[iPMT] = pmt.GetOrientation(0)
    out_pmt_py[iPMT] = pmt.GetOrientation(1)
    out_pmt_pz[iPMT] = pmt.GetOrientation(2)

print("@@@ Start analysing data")
out_vtx_x = np.zeros(nEvents)
out_vtx_y = np.zeros(nEvents)
out_vtx_z = np.zeros(nEvents)
out_vtx_t = np.zeros(nEvents)

out_vtx_dx = np.zeros(nEvents)
out_vtx_dy = np.zeros(nEvents)
out_vtx_dz = np.zeros(nEvents)

out_vtx_px = np.zeros(nEvents)
out_vtx_py = np.zeros(nEvents)
out_vtx_pz = np.zeros(nEvents)
out_vtx_ke = np.zeros(nEvents)
out_vtx_ke2 = np.zeros(nEvents)

out_pmt_q = np.zeros((nEvents, nPMTs))
out_pmt_t = np.zeros((nEvents, nPMTs))
for iEvent in tqdm(range(nEvents)):
    eventT.GetEvent(iEvent)
    trigger = event.GetTrigger(0)

    if trigger.GetNvtxs() == 0: continue
    if trigger.GetNtrack() == 0: continue

    out_vtx_x[iEvent] = trigger.GetVtx(0)
    out_vtx_y[iEvent] = trigger.GetVtx(1)
    out_vtx_z[iEvent] = trigger.GetVtx(2)
    out_vtx_t[iEvent] = 0

    firstTrack = trigger.GetTracks()[0]

    out_vtx_dx[iEvent] = firstTrack.GetDir(0)
    out_vtx_dy[iEvent] = firstTrack.GetDir(1)
    out_vtx_dz[iEvent] = firstTrack.GetDir(2)

    out_vtx_px[iEvent] = firstTrack.GetPdir(0)
    out_vtx_py[iEvent] = firstTrack.GetPdir(1)
    out_vtx_pz[iEvent] = firstTrack.GetPdir(2)
    out_vtx_ke[iEvent] = firstTrack.GetE()
    out_vtx_ke2[iEvent] = firstTrack.GetE()-firstTrack.GetM()
    nHitsC = trigger.GetNcherenkovdigihits()
    for iHit in range(nHitsC):
        hit = trigger.GetCherenkovDigiHits().At(iHit)
        iPMT = hit.GetTubeId()-1
        out_pmt_q[iEvent, iPMT] = hit.GetQ()
        out_pmt_t[iEvent, iPMT] = hit.GetT()

results = []
stop_vtx_x = []
stop_vtx_y = []
stop_vtx_z = []
nEntries = eventT.GetEntries()
for i in range(nEntries):
    eventT.GetEntry(i)

    trigger = event.GetTrigger(0)


    vtx_x = trigger.GetVtx(0)
    vtx_y = trigger.GetVtx(1)
    vtx_z = trigger.GetVtx(2)


    nTracks = trigger.GetNtrack()


    primary_track = None
    primary_pdg_code = None  
    primary_track_id = None 

    for j in range(nTracks):
        track = trigger.GetTracks().At(j)
        parent_id = track.GetParenttype()  # Parent ID
        pdg_code = track.GetIpnu()         # PDG Code

        # Parent ID가 0인 경우 주 입자
        if parent_id == 0:
            primary_track = track
            primary_pdg_code = pdg_code    
            primary_track_id = track.GetId()  
            break

    if primary_track is None:
        print(f"Event {i}: Primary track not found.")
        continue


    initial_energy = primary_track.GetE()  # MeV 단위

    secondary_tracks = []

    for j in range(nTracks):
        track = trigger.GetTracks().At(j)
        parent_id = track.GetParenttype()  # Parent ID


        if parent_id == primary_track_id and track != primary_track:
            secondary_tracks.append(track)


    for track in secondary_tracks:
        pdg_code = track.GetIpnu()
        x_start = track.GetStart(0)
        y_start = track.GetStart(1)
        z_start = track.GetStart(2)

        x_stop = track.GetStop(0)
        y_stop = track.GetStop(1)
        z_stop = track.GetStop(2)

        energy = track.GetE()  # 입자의 에너지


    final_energy = 0.0

    n_digi_hits = trigger.GetNcherenkovdigihits()
    digi_hits = trigger.GetCherenkovDigiHits()

    total_charge = 0.0

    for j in range(n_digi_hits):
        digi_hit = digi_hits.At(j)
        charge = digi_hit.GetQ()
        total_charge += charge


    calibration_factor = 0.1  

    total_detected_energy = total_charge * calibration_factor  


    result = {
        'Event': i,
        'InitialEnergy': initial_energy,
        'FinalEnergy': final_energy,
        'EnergyDepositedInPMTs': total_detected_energy,
        'EnergyDifference': initial_energy - (final_energy + total_detected_energy),
        'StartPosition': (x_start, y_start, z_start),
        'StopPosition': (x_stop, y_stop, z_stop),
        'VertexPosition': (vtx_x, vtx_y, vtx_z)
    }
    stop_vtx_x.append(x_stop)
    stop_vtx_y.append(y_stop)
    stop_vtx_z.append(z_stop)
    results.append(result)



positions = np.stack((out_pmt_x, out_pmt_y, out_pmt_z), axis=1)
events = np.stack((out_vtx_x, out_vtx_y, out_vtx_z), axis=1)
end_events = np.stack((np.array(stop_vtx_x),np.array(stop_vtx_y),np.array(stop_vtx_z)), axis=1)


differences = events[:, np.newaxis, :] - positions[np.newaxis, :, :]
differences_end = end_events[:, np.newaxis, :] - positions[np.newaxis, :, :]
differences_start_end = events - end_events


squared_differences = differences ** 2
squared_differences_end = differences_end ** 2
squared_differences_start_end = differences_start_end ** 2

sum_squared_differences = np.sum(squared_differences, axis=2)
sum_squared_differences_end = np.sum(squared_differences_end, axis=2)
sum_squared_differences_start_end = np.sum(squared_differences_start_end, axis=1)

distances = np.sqrt(sum_squared_differences)  # 형태: (2000, 30912)
distances_end = np.sqrt(sum_squared_differences_end)  # 형태: (2000, 30912)
distances_start_end = np.sqrt(sum_squared_differences_start_end)  # 형태: (2000, )


min_distances = np.min(distances, axis=1)  # 형태: (2000,)
min_indices = np.argmin(distances, axis=1)  # 형태: (2000,)
min_distances_end = np.min(distances_end, axis=1)  # 형태: (2000,)
min_indices_end = np.argmin(distances_end, axis=1)  # 형태: (2000,)


pmts_num = (np.sum(out_pmt_q>0,axis=1) > 1000)

aaa = min_distances > 200




out_vtx_x = out_vtx_x[pmts_num & aaa]
out_vtx_y = out_vtx_y[pmts_num & aaa]
out_vtx_z = out_vtx_z[pmts_num & aaa]
out_vtx_t = out_vtx_t[pmts_num & aaa]

out_vtx_px = out_vtx_px[pmts_num & aaa]
out_vtx_py = out_vtx_py[pmts_num & aaa]
out_vtx_pz = out_vtx_pz[pmts_num & aaa]
out_vtx_ke = out_vtx_ke[pmts_num & aaa]
out_vtx_ke2 = out_vtx_ke2[pmts_num & aaa]

out_pmt_t = out_pmt_t[pmts_num & aaa]
out_pmt_q = out_pmt_q[pmts_num & aaa]
min_distances = min_distances[pmts_num & aaa]
min_distances_end = min_distances_end[pmts_num & aaa]
distances_start_end = distances_start_end[pmts_num & aaa]


stop_vtx_x = np.array(stop_vtx_x)[pmts_num & aaa]
stop_vtx_y = np.array(stop_vtx_y)[pmts_num & aaa]
stop_vtx_z = np.array(stop_vtx_z)[pmts_num & aaa]


cut_events_test = (out_pmt_q.shape[0]/2000)*100
print("cut % = "+ f"{cut_events_test:.3f}"+"%")



if out_pmt_q.shape[0] > 0:
    kwargs = {'dtype':'f4', 'compression':'lzf'}

    with h5py.File(args.output, 'w', libver='latest') as fout:
        gGeom = fout.create_group('geom')
        gGeom.create_dataset('pmt_x', data=out_pmt_x, **kwargs)
        gGeom.create_dataset('pmt_y', data=out_pmt_y, **kwargs)
        gGeom.create_dataset('pmt_z', data=out_pmt_z, **kwargs)

        gGeom.create_dataset('pmt_px', data=out_pmt_px, **kwargs)
        gGeom.create_dataset('pmt_py', data=out_pmt_py, **kwargs)
        gGeom.create_dataset('pmt_pz', data=out_pmt_pz, **kwargs)

        gGeom.create_dataset('stop_x', data=stop_vtx_x, **kwargs)
        gGeom.create_dataset('stop_y', data=stop_vtx_y, **kwargs)
        gGeom.create_dataset('stop_z', data=stop_vtx_z, **kwargs)

        gEvent = fout.create_group('event')
        gEvent.create_dataset('vtx_x', data=out_vtx_x, **kwargs)
        gEvent.create_dataset('vtx_y', data=out_vtx_y, **kwargs)
        gEvent.create_dataset('vtx_z', data=out_vtx_z, **kwargs)
        gEvent.create_dataset('vtx_t', data=out_vtx_t, **kwargs)

        gEvent.create_dataset('distances', data=min_distances, **kwargs)
        gEvent.create_dataset('distances_end', data=min_distances_end, **kwargs)
        gEvent.create_dataset('distances_start_end', data=distances_start_end, **kwargs)

        gEvent.create_dataset('vtx_px', data=out_vtx_px, **kwargs)
        gEvent.create_dataset('vtx_py', data=out_vtx_py, **kwargs)
        gEvent.create_dataset('vtx_pz', data=out_vtx_pz, **kwargs)
        gEvent.create_dataset('vtx_ke', data=out_vtx_ke, **kwargs)
        gEvent.create_dataset('vtx_ke2', data=out_vtx_ke2, **kwargs)

        gEvent.create_dataset('pmt_q', data=out_pmt_q, **kwargs)
        gEvent.create_dataset('pmt_t', data=out_pmt_t, **kwargs)