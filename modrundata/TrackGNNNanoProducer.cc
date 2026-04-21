#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

class TrackGNNNanoProducer : public edm::stream::EDProducer<edm::GlobalCache<cms::Ort::ONNXRuntime>> {
public:
  explicit TrackGNNNanoProducer(const edm::ParameterSet& cfg, const cms::Ort::ONNXRuntime* cache)
      : onnxRuntime_(cache),
        tracksToken_(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("tracks"))),
        lostTracksToken_(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("losttracks"))),
        pvToken_(consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("primaryVertices"))),
        ttbToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
        trackPtCut_(cfg.getParameter<double>("trackPtCut")),
        requireHighPurity_(cfg.getParameter<bool>("requireHighPurity")),
        requireTrackDetails_(cfg.getParameter<bool>("requireTrackDetails")),
        edgeDeltaRMin_(cfg.getParameter<double>("edgeDeltaRMin")),
        edgeDeltaRMax_(cfg.getParameter<double>("edgeDeltaRMax")),
        edgeInvMassMax_(cfg.getParameter<double>("edgeInvMassMax")),
        edgeDcaMin_(cfg.getParameter<double>("edgeDcaMin")),
        edgeDcaMax_(cfg.getParameter<double>("edgeDcaMax")),
        edgeDcaSigMax_(cfg.getParameter<double>("edgeDcaSigMax")),
        edgeCpToPvMin_(cfg.getParameter<double>("edgeCpToPvMin")),
        edgeCpToPvMax_(cfg.getParameter<double>("edgeCpToPvMax")),
        edgePvToPcaMax_(cfg.getParameter<double>("edgePvToPcaMax")),
        edgePairMomMin_(cfg.getParameter<double>("edgePairMomMin")),
        edgePairMomMax_(cfg.getParameter<double>("edgePairMomMax")),
        writeLogits_(cfg.getParameter<bool>("writeLogits")) {
    produces<nanoaod::FlatTable>("TrackGNNTrackTable");
    produces<nanoaod::FlatTable>("TrackGNNEdgeTable");
    produces<nanoaod::FlatTable>("TrackGNNSummaryTable");
  }

  static std::unique_ptr<cms::Ort::ONNXRuntime> initializeGlobalCache(const edm::ParameterSet& cfg) {
    return std::make_unique<cms::Ort::ONNXRuntime>(cfg.getParameter<edm::FileInPath>("model_path").fullPath());
  }

  static void globalEndJob(const cms::Ort::ONNXRuntime*) {}

  void produce(edm::Event& event, const edm::EventSetup& setup) override {
    std::vector<float> trackSVNonSV;
    std::vector<float> trackSVPV;
    std::vector<float> trackSVSV;
    std::vector<float> trackSub0;
    std::vector<float> trackSub1;
    std::vector<float> trackSub2;

    std::vector<float> svLogit0;
    std::vector<float> svLogit1;
    std::vector<float> svLogit2;
    std::vector<float> subLogit0;
    std::vector<float> subLogit1;
    std::vector<float> subLogit2;

    std::vector<int> edgeSrc;
    std::vector<int> edgeDst;
    std::vector<float> edgeProb;
    std::vector<float> edgeLogit;

    unsigned int nTrackOut = 0;
    unsigned int nEdgeOut = 0;
    unsigned int nEdgeRawOut = 0;

    const auto& tracks = event.get(tracksToken_);
    const auto& lostTracks = event.get(lostTracksToken_);
    const auto& pvs = event.get(pvToken_);

    if (pvs.empty()) {
      putOutputs(event,
                 nTrackOut,
                 nEdgeOut,
                 nEdgeRawOut,
                 trackSVNonSV,
                 trackSVPV,
                 trackSVSV,
                 trackSub0,
                 trackSub1,
                 trackSub2,
                 edgeSrc,
                 edgeDst,
                 edgeProb,
                 svLogit0,
                 svLogit1,
                 svLogit2,
                 subLogit0,
                 subLogit1,
                 subLogit2,
                 edgeLogit);
      return;
    }

    const reco::Vertex& pv = pvs.front();
    const auto& ttb = setup.getData(ttbToken_);

    std::vector<reco::Track> allTracks;
    std::vector<reco::TransientTrack> transientTracks;

    auto addPacked = [&](const pat::PackedCandidateCollection& collection) {
      for (const auto& cand : collection) {
        if (requireTrackDetails_ && !cand.hasTrackDetails())
          continue;
        if (!cand.hasTrackDetails())
          continue;

        reco::Track trk = cand.pseudoTrack();
        if (trk.charge() == 0)
          continue;
        if (trk.pt() < trackPtCut_)
          continue;

        if (requireHighPurity_) {
          if (!cand.trackHighPurity())
            continue;
          if (!trk.quality(reco::TrackBase::highPurity))
            continue;
        }

        allTracks.push_back(trk);
        transientTracks.emplace_back(ttb.build(trk));
      }
    };

    addPacked(tracks);
    addPacked(lostTracks);

    const size_t nTracks = allTracks.size();
    nTrackOut = static_cast<unsigned int>(nTracks);

    if (nTracks == 0) {
      putOutputs(event,
                 nTrackOut,
                 nEdgeOut,
                 nEdgeRawOut,
                 trackSVNonSV,
                 trackSVPV,
                 trackSVSV,
                 trackSub0,
                 trackSub1,
                 trackSub2,
                 edgeSrc,
                 edgeDst,
                 edgeProb,
                 svLogit0,
                 svLogit1,
                 svLogit2,
                 subLogit0,
                 subLogit1,
                 subLogit2,
                 edgeLogit);
      return;
    }

    std::vector<std::array<float, 14>> trackFeatures;
    trackFeatures.reserve(nTracks);

    for (size_t i = 0; i < nTracks; ++i) {
      const auto& trk = allTracks[i];
      const auto& tt = transientTracks[i];

      float ip2d = -999.f;
      float ip3d = -999.f;
      float ip2dsig = -999.f;
      float ip3dsig = -999.f;

      auto ip2dMeas = IPTools::absoluteTransverseImpactParameter(tt, pv);
      if (ip2dMeas.first && std::isfinite(ip2dMeas.second.value())) {
        ip2d = std::abs(ip2dMeas.second.value());
        ip2dsig = std::abs(ip2dMeas.second.significance());
      }

      auto ip3dMeas = IPTools::absoluteImpactParameter3D(tt, pv);
      if (ip3dMeas.first && std::isfinite(ip3dMeas.second.value())) {
        ip3d = std::abs(ip3dMeas.second.value());
        ip3dsig = std::abs(ip3dMeas.second.significance());
      }

      const float dz = std::abs(trk.dz(pv.position()));
      const float dzsig = (trk.dzError() > 0.f) ? std::abs(trk.dz(pv.position()) / trk.dzError()) : -999.f;

      trackFeatures.push_back({{static_cast<float>(trk.eta()),
                                static_cast<float>(trk.phi()),
                                ip2d,
                                ip3d,
                                dz,
                                dzsig,
                                ip2dsig,
                                ip3dsig,
                                static_cast<float>(trk.p()),
                                static_cast<float>(trk.pt()),
                                static_cast<float>(trk.numberOfValidHits()),
                                static_cast<float>(trk.hitPattern().numberOfValidPixelHits()),
                                static_cast<float>(trk.hitPattern().numberOfValidStripHits()),
                                static_cast<float>(trk.charge())}});
    }

    std::vector<int> localSrc;
    std::vector<int> localDst;
    std::vector<std::array<float, 10>> edgeFeatures;

    localSrc.reserve(nTracks * (nTracks - 1) / 2);
    localDst.reserve(nTracks * (nTracks - 1) / 2);
    edgeFeatures.reserve(nTracks * (nTracks - 1) / 2);

    constexpr float pionMass = 0.13957018f;
    for (size_t i = 0; i < nTracks; ++i) {
      if (!transientTracks[i].isValid())
        continue;
      for (size_t j = i + 1; j < nTracks; ++j) {
        if (!transientTracks[j].isValid())
          continue;

        ++nEdgeRawOut;

        const float deltaR = reco::deltaR(allTracks[i].eta(), allTracks[i].phi(), allTracks[j].eta(), allTracks[j].phi());
        if (deltaR < edgeDeltaRMin_ || deltaR > edgeDeltaRMax_)
          continue;

        const float px1 = allTracks[i].px();
        const float py1 = allTracks[i].py();
        const float pz1 = allTracks[i].pz();
        const float px2 = allTracks[j].px();
        const float py2 = allTracks[j].py();
        const float pz2 = allTracks[j].pz();

        const float e1 = std::sqrt(px1 * px1 + py1 * py1 + pz1 * pz1 + pionMass * pionMass);
        const float e2 = std::sqrt(px2 * px2 + py2 * py2 + pz2 * pz2 + pionMass * pionMass);
        const float sumPx = px1 + px2;
        const float sumPy = py1 + py2;
        const float sumPz = pz1 + pz2;
        const float sumE = e1 + e2;
        const float invMass2 = sumE * sumE - (sumPx * sumPx + sumPy * sumPy + sumPz * sumPz);
        const float invMass = (invMass2 > 0.f) ? std::sqrt(invMass2) : 0.f;
        if (invMass > edgeInvMassMax_)
          continue;

        float dca = -1.f;
        float dcaSig = -1.f;
        float cpToPv = -1.f;
        float pvToPcaI = -1.f;
        float pvToPcaJ = -1.f;
        float dotI = -999.f;
        float dotJ = -999.f;
        float pairMom = -1.f;

        TwoTrackMinimumDistance minDist;
        if (!minDist.calculate(transientTracks[i].impactPointState(), transientTracks[j].impactPointState()))
          continue;

        VertexDistance3D distanceComputer;
        auto d3d = distanceComputer.distance(
            VertexState(minDist.points().second, transientTracks[i].impactPointState().cartesianError().position()),
            VertexState(minDist.points().first, transientTracks[j].impactPointState().cartesianError().position()));

        dca = d3d.value();
        if (d3d.error() > 0.f)
          dcaSig = d3d.value() / d3d.error();

        const GlobalPoint cp(minDist.crossingPoint());
        const GlobalPoint pvp(pv.x(), pv.y(), pv.z());
        const GlobalPoint seedPca = minDist.points().second;
        const GlobalPoint trackPca = minDist.points().first;

        cpToPv = (cp - pvp).mag();
        pvToPcaI = (seedPca - pvp).mag();
        pvToPcaJ = (trackPca - pvp).mag();

        dotI = (seedPca - pvp).unit().dot(transientTracks[i].impactPointState().globalDirection().unit());
        dotJ = (trackPca - pvp).unit().dot(transientTracks[j].impactPointState().globalDirection().unit());

        GlobalVector pairMomentum((Basic3DVector<float>)(allTracks[i].momentum() + allTracks[j].momentum()));
        pairMom = pairMomentum.mag();

        if (dca < edgeDcaMin_ || dca > edgeDcaMax_)
          continue;
        if (dcaSig > edgeDcaSigMax_)
          continue;
        if (cpToPv < edgeCpToPvMin_ || cpToPv > edgeCpToPvMax_)
          continue;
        if (pvToPcaI > edgePvToPcaMax_ || pvToPcaJ > edgePvToPcaMax_)
          continue;
        if (pairMom < edgePairMomMin_ || pairMom > edgePairMomMax_)
          continue;

        localSrc.push_back(static_cast<int>(i));
        localDst.push_back(static_cast<int>(j));
        edgeFeatures.push_back({{dca, deltaR, dcaSig, cpToPv, pvToPcaI, pvToPcaJ, dotI, dotJ, pairMom, invMass}});
      }
    }

    const size_t nEdges = localSrc.size();
    nEdgeOut = static_cast<unsigned int>(nEdges);

    if (nEdges == 0) {
      putOutputs(event,
                 nTrackOut,
                 nEdgeOut,
                 nEdgeRawOut,
                 trackSVNonSV,
                 trackSVPV,
                 trackSVSV,
                 trackSub0,
                 trackSub1,
                 trackSub2,
                 edgeSrc,
                 edgeDst,
                 edgeProb,
                 svLogit0,
                 svLogit1,
                 svLogit2,
                 subLogit0,
                 subLogit1,
                 subLogit2,
                 edgeLogit);
      return;
    }

    std::vector<float> xInFlat;
    xInFlat.reserve(nTracks * 14);
    for (const auto& f : trackFeatures)
      xInFlat.insert(xInFlat.end(), f.begin(), f.end());

    std::vector<float> edgeIndexFlat;
    edgeIndexFlat.reserve(nEdges * 2);
    for (const auto& s : localSrc)
      edgeIndexFlat.push_back(static_cast<float>(s));
    for (const auto& d : localDst)
      edgeIndexFlat.push_back(static_cast<float>(d));

    std::vector<float> edgeAttrFlat;
    edgeAttrFlat.reserve(nEdges * 10);
    for (const auto& f : edgeFeatures)
      edgeAttrFlat.insert(edgeAttrFlat.end(), f.begin(), f.end());

    std::vector<std::string> inputNames = {"x_in", "edge_index", "edge_attr"};
    std::vector<std::vector<int64_t>> inputShapes = {
        {1, static_cast<int64_t>(nTracks), 14},
        {1, 2, static_cast<int64_t>(nEdges)},
        {1, static_cast<int64_t>(nEdges), 10},
    };
    std::vector<std::vector<float>> inputData = {xInFlat, edgeIndexFlat, edgeAttrFlat};

    const auto output = onnxRuntime_->run(inputNames, inputData, inputShapes);
    if (output.size() < 4)
      throw cms::Exception("TrackGNNNanoProducer") << "Model output has fewer than 4 tensors.";

    const auto& svLogits = output[0];
    const auto& subLogits = output[1];
    const auto& edgeLogits = output[2];

    if (svLogits.size() != nTracks * 3 || subLogits.size() != nTracks * 3 || edgeLogits.size() != nEdges) {
      throw cms::Exception("TrackGNNNanoProducer") << "Unexpected model output sizes: "
                                                    << "sv=" << svLogits.size() << ", sub=" << subLogits.size()
                                                    << ", edge=" << edgeLogits.size() << ", N=" << nTracks
                                                    << ", E=" << nEdges;
    }

    trackSVNonSV.reserve(nTracks);
    trackSVPV.reserve(nTracks);
    trackSVSV.reserve(nTracks);
    trackSub0.reserve(nTracks);
    trackSub1.reserve(nTracks);
    trackSub2.reserve(nTracks);

    if (writeLogits_) {
      svLogit0.reserve(nTracks);
      svLogit1.reserve(nTracks);
      svLogit2.reserve(nTracks);
      subLogit0.reserve(nTracks);
      subLogit1.reserve(nTracks);
      subLogit2.reserve(nTracks);
      edgeLogit.reserve(nEdges);
    }

    for (size_t i = 0; i < nTracks; ++i) {
      float pNode[3];
      softmax3(svLogits[i * 3], svLogits[i * 3 + 1], svLogits[i * 3 + 2], pNode);
      trackSVNonSV.push_back(pNode[0]);
      trackSVPV.push_back(pNode[1]);
      trackSVSV.push_back(pNode[2]);

      float pSub[3];
      softmax3(subLogits[i * 3], subLogits[i * 3 + 1], subLogits[i * 3 + 2], pSub);
      trackSub0.push_back(pSub[0]);
      trackSub1.push_back(pSub[1]);
      trackSub2.push_back(pSub[2]);

      if (writeLogits_) {
        svLogit0.push_back(svLogits[i * 3]);
        svLogit1.push_back(svLogits[i * 3 + 1]);
        svLogit2.push_back(svLogits[i * 3 + 2]);
        subLogit0.push_back(subLogits[i * 3]);
        subLogit1.push_back(subLogits[i * 3 + 1]);
        subLogit2.push_back(subLogits[i * 3 + 2]);
      }
    }

    edgeSrc = localSrc;
    edgeDst = localDst;
    edgeProb.reserve(nEdges);
    for (size_t e = 0; e < nEdges; ++e) {
      edgeProb.push_back(sigmoid(edgeLogits[e]));
      if (writeLogits_)
        edgeLogit.push_back(edgeLogits[e]);
    }

    if (trackSVNonSV.size() != nTracks || trackSVPV.size() != nTracks || trackSVSV.size() != nTracks ||
    trackSub0.size() != nTracks || trackSub1.size() != nTracks || trackSub2.size() != nTracks) {
  throw cms::Exception("TrackGNNNanoProducer")
      << "Track output size mismatch: "
      << "nTracks=" << nTracks
      << " svNonSV=" << trackSVNonSV.size()
      << " svPV=" << trackSVPV.size()
      << " svSV=" << trackSVSV.size()
      << " sub0=" << trackSub0.size()
      << " sub1=" << trackSub1.size()
      << " sub2=" << trackSub2.size();
}

if (edgeSrc.size() != nEdges || edgeDst.size() != nEdges || edgeProb.size() != nEdges) {
  throw cms::Exception("TrackGNNNanoProducer")
      << "Edge output size mismatch: "
      << "nEdges=" << nEdges
      << " src=" << edgeSrc.size()
      << " dst=" << edgeDst.size()
      << " prob=" << edgeProb.size();
}

if(writeLogits_) {
	if (svLogit0.size() != nTracks || svLogit1.size() != nTracks || svLogit2.size() != nTracks ||
    subLogit0.size() != nTracks || subLogit1.size() != nTracks || subLogit2.size() != nTracks ||
    edgeLogit.size() != nEdges) {
  throw cms::Exception("TrackGNNNanoProducer")
      << "Logit output size mismatch.";
	}
}

    putOutputs(event,
               nTrackOut,
               nEdgeOut,
               nEdgeRawOut,
               trackSVNonSV,
               trackSVPV,
               trackSVSV,
               trackSub0,
               trackSub1,
               trackSub2,
               edgeSrc,
               edgeDst,
               edgeProb,
               svLogit0,
               svLogit1,
               svLogit2,
               subLogit0,
               subLogit1,
               subLogit2,
               edgeLogit);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("tracks", edm::InputTag("packedPFCandidates"));
    desc.add<edm::InputTag>("losttracks", edm::InputTag("lostTracks", "", "PAT"));
    desc.add<edm::InputTag>("primaryVertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
    desc.add<edm::FileInPath>("model_path", edm::FileInPath("nano-custom-trackGNN/models/bhive_newsubmod_out128_1404.onnx"));

    desc.add<double>("trackPtCut", 0.0);
    desc.add<bool>("requireHighPurity", false);
    desc.add<bool>("requireTrackDetails", false);

    desc.add<double>("edgeDeltaRMin", 2e-4);
    desc.add<double>("edgeDeltaRMax", 1.0);
    desc.add<double>("edgeInvMassMax", 20.0);
    desc.add<double>("edgeDcaMin", 1e-8);
    desc.add<double>("edgeDcaMax", 1.0);
    desc.add<double>("edgeDcaSigMax", 100.0);
    desc.add<double>("edgeCpToPvMin", 4e-4);
    desc.add<double>("edgeCpToPvMax", 20.0);
    desc.add<double>("edgePvToPcaMax", 20.0);
    desc.add<double>("edgePairMomMin", 0.05);
    desc.add<double>("edgePairMomMax", 100.0);

    desc.add<bool>("writeLogits", false);

    descriptions.addDefault(desc);

  }

private:
  static inline float sigmoid(float x) {
    if (x >= 0.f) {
      const float z = std::exp(-x);
      return 1.f / (1.f + z);
    }
    const float z = std::exp(x);
    return z / (1.f + z);
  }

  static inline void softmax3(float a, float b, float c, float out[3]) {
    const float m = std::max(a, std::max(b, c));
    const float ea = std::exp(a - m);
    const float eb = std::exp(b - m);
    const float ec = std::exp(c - m);
    const float s = ea + eb + ec;
    out[0] = ea / s;
    out[1] = eb / s;
    out[2] = ec / s;
  }

  void putOutputs(edm::Event& event,
                  unsigned int nTrackOut,
                  unsigned int nEdgeOut,
                  unsigned int nEdgeRawOut,
                  const std::vector<float>& trackSVNonSV,
                  const std::vector<float>& trackSVPV,
                  const std::vector<float>& trackSVSV,
                  const std::vector<float>& trackSub0,
                  const std::vector<float>& trackSub1,
                  const std::vector<float>& trackSub2,
                  const std::vector<int>& edgeSrc,
                  const std::vector<int>& edgeDst,
                  const std::vector<float>& edgeProb,
                  const std::vector<float>& svLogit0,
                  const std::vector<float>& svLogit1,
                  const std::vector<float>& svLogit2,
                  const std::vector<float>& subLogit0,
                  const std::vector<float>& subLogit1,
                  const std::vector<float>& subLogit2,
                  const std::vector<float>& edgeLogit) const {
    auto summary = std::make_unique<nanoaod::FlatTable>(1, "TrackGNNSummary", true, false);
    summary->addColumnValue<int>("nTrack", static_cast<int>(nTrackOut), "Number of TrackGNN input tracks");
    summary->addColumnValue<int>("nEdge", static_cast<int>(nEdgeOut), "Number of accepted TrackGNN edges");
    summary->addColumnValue<int>("nEdgeRaw", static_cast<int>(nEdgeRawOut), "Number of raw edge pairs before final selection");
    event.put(std::move(summary), "TrackGNNSummaryTable");

    auto trackTable = std::make_unique<nanoaod::FlatTable>(trackSVNonSV.size(), "TrackGNN", false, false);
    trackTable->addColumn<float>("svProbNonSV", trackSVNonSV, "Track-level SV class probability: nonSV");
    trackTable->addColumn<float>("svProbPV", trackSVPV, "Track-level SV class probability: PV");
    trackTable->addColumn<float>("svProbSV", trackSVSV, "Track-level SV class probability: SV");
    trackTable->addColumn<float>("subProb0", trackSub0, "Track-level subclass probability 0");
    trackTable->addColumn<float>("subProb1", trackSub1, "Track-level subclass probability 1");
    trackTable->addColumn<float>("subProb2", trackSub2, "Track-level subclass probability 2");
    if (writeLogits_) {
      trackTable->addColumn<float>("svLogitNonSV", svLogit0, "Track-level SV logit: nonSV");
      trackTable->addColumn<float>("svLogitPV", svLogit1, "Track-level SV logit: PV");
      trackTable->addColumn<float>("svLogitSV", svLogit2, "Track-level SV logit: SV");
      trackTable->addColumn<float>("subLogit0", subLogit0, "Track-level subclass logit 0");
      trackTable->addColumn<float>("subLogit1", subLogit1, "Track-level subclass logit 1");
      trackTable->addColumn<float>("subLogit2", subLogit2, "Track-level subclass logit 2");
    }
    event.put(std::move(trackTable), "TrackGNNTrackTable");

    auto edgeTable = std::make_unique<nanoaod::FlatTable>(edgeProb.size(), "TrackGNNEdge", false, false);
    edgeTable->addColumn<int>("src", edgeSrc, "Source track index into TrackGNN table");
    edgeTable->addColumn<int>("dst", edgeDst, "Destination track index into TrackGNN table");
    edgeTable->addColumn<float>("prob", edgeProb, "Edge probability");
    if (writeLogits_) {
      edgeTable->addColumn<float>("logit", edgeLogit, "Edge logit");
    }
    event.put(std::move(edgeTable), "TrackGNNEdgeTable");
  }

  const cms::Ort::ONNXRuntime* onnxRuntime_;
  edm::EDGetTokenT<pat::PackedCandidateCollection> tracksToken_;
  edm::EDGetTokenT<pat::PackedCandidateCollection> lostTracksToken_;
  edm::EDGetTokenT<reco::VertexCollection> pvToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttbToken_;

  double trackPtCut_;
  bool requireHighPurity_;
  bool requireTrackDetails_;

  double edgeDeltaRMin_;
  double edgeDeltaRMax_;
  double edgeInvMassMax_;
  double edgeDcaMin_;
  double edgeDcaMax_;
  double edgeDcaSigMax_;
  double edgeCpToPvMin_;
  double edgeCpToPvMax_;
  double edgePvToPcaMax_;
  double edgePairMomMin_;
  double edgePairMomMax_;

  bool writeLogits_;
};

DEFINE_FWK_MODULE(TrackGNNNanoProducer);

