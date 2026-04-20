import FWCore.ParameterSet.Config as cms

trackGNNNano = cms.EDProducer(
    "TrackGNNNanoProducer",
    # Inputs
    tracks=cms.InputTag("packedPFCandidates"),
    losttracks=cms.InputTag("lostTracks", "", "PAT"),
    primaryVertices=cms.InputTag("offlineSlimmedPrimaryVertices"),

    # Model
    model_path=cms.FileInPath("dispV/dispVAnalyzer/data/bhive_hcmod_1703.onnx"),

    # Track handling (all tracks by default)
    trackPtCut=cms.double(0.0),
    requireHighPurity=cms.bool(False),
    requireTrackDetails=cms.bool(False),

    # Edge feature preselection (only edge control; no post-inference gates)
    edgeDeltaRMin=cms.double(2e-4),
    edgeDeltaRMax=cms.double(1.0),
    edgeInvMassMax=cms.double(20.0),
    edgeDcaMin=cms.double(1e-8),
    edgeDcaMax=cms.double(1.0),
    edgeDcaSigMax=cms.double(100.0),
    edgeCpToPvMin=cms.double(4e-4),
    edgeCpToPvMax=cms.double(20.0),
    edgePvToPcaMax=cms.double(20.0),
    edgePairMomMin=cms.double(0.05),
    edgePairMomMax=cms.double(100.0),

    # Output switches
    writeTrackSVProb=cms.bool(True),
    writeTrackSubProb=cms.bool(True),
    writeEdgeProb=cms.bool(True),
    writeEdgeIndex=cms.bool(True),
    writeLogits=cms.bool(False),
)
