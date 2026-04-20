import FWCore.ParameterSet.Config as cms

from modrundata.trackGNN_cfi import trackGNNNano


def nanoAOD_addTrackGNN(process):
    process.trackGNNNano = trackGNNNano

    if hasattr(process, "nanoSequence"):
        process.nanoSequence = cms.Sequence(process.nanoSequence + process.trackGNNNano)
    else:
        process.trackGNN_step = cms.Path(process.trackGNNNano)

    if hasattr(process, "NANOAODoutput") and hasattr(process.NANOAODoutput, "outputCommands"):
        process.NANOAODoutput.outputCommands.extend([
            "keep *_trackGNNNano_*_*",
            "keep nanoaodFlatTable_trackGNNNano*_*_*",
        ])

    return process
