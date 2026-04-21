import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.trackGNN_cfi import trackGNNNano

def nanoAOD_addTrackGNN(process):
    process.trackGNNNano = trackGNNNano.clone()

    if hasattr(process, "nanoSequenceMC"):
        process.nanoSequenceMC = cms.Sequence(process.nanoSequenceMC + process.trackGNNNano)
    if hasattr(process, "nanoSequence"):
        process.nanoSequence = cms.Sequence(process.nanoSequence + process.trackGNNNano)

    if not hasattr(process, "nanoSequenceMC") and not hasattr(process, "nanoSequence"):
        process.trackGNN_step = cms.Path(process.trackGNNNano)

    keep_cmds = [
        "keep nanoaodFlatTable_trackGNNNano_TrackGNNTrackTable_*",
        "keep nanoaodFlatTable_trackGNNNano_TrackGNNEdgeTable_*",
        "keep nanoaodFlatTable_trackGNNNano_TrackGNNSummaryTable_*",
    ]

    for out in ("NANOAODoutput", "NANOAODSIMoutput"):
        if hasattr(process, out) and hasattr(getattr(process, out), "outputCommands"):
            getattr(process, out).outputCommands.extend(keep_cmds)

    return process
