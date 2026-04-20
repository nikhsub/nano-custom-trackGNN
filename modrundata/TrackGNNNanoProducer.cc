#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoVertex/VertexPrimitives/interface/VertexDistance3D.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

class TrackGNNNanoProducer : public edm::global::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit TrackGNNNanoProducer(const edm::ParameterSet &cfg)
      : tracksToken_(consumes<pat::PackedCandidateCollection>(cfg.getParameter<edm::InputTag>("tracks"))),
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
    produces<unsigned int>("nTrackGNNTrack");
    produces<unsigned int>("nTrackGNNEdge");
    produces<unsigned int>("nTrackGNNEdgeRaw");

    produces<std::vector<float>>("TrackGNN_track_SV_prob_nonSV");
    produces<std::vector<float>>("TrackGNN_track_SV_prob_PV");
    produces<std::vector<float>>("TrackGNN_track_SV_prob_SV");

    produces<std::vector<float>>("TrackGNN_track_sub_prob_0");
    produces<std::vector<float>>("TrackGNN_track_sub_prob_1");
    produces<std::vector<float>>("TrackGNN_track_sub_prob_2");

    produces<std::vector<int>>("TrackGNN_edge_src");
    produces<std::vector<int>>("TrackGNN_edge_dst");
    produces<std::vector<float>>("TrackGNN_edge_prob");

    if (writeLogits_) {
      produces<std::vector<float>>("TrackGNN_track_SV_logit_nonSV");
      produces<std::vector<float>>("TrackGNN_track_SV_logit_PV");
      produces<std::vector<float>>("TrackGNN_track_SV_logit_SV");
      produces<std::vector<float>>("TrackGNN_track_sub_logit_0");
      produces<std::vector<float>>("TrackGNN_track_sub_logit_1");
      produces<std::vector<float>>("TrackGNN_track_sub_logit_2");
      produces<std::vector<float>>("TrackGNN_edge_logit");
    }
  }

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet &cfg) {
    return std::make_unique<ONNXRuntime>(cfg.getParameter<edm::FileInPath>("model_path").fullPath());
  }

  static void globalEndJob(const ONNXRuntime *) {}

  void produce(edm::StreamID, edm::Event &event, const edm::EventSetup &setup) const override {
    auto nTrackOut = std::make_unique<unsigned int>(0);
    auto nEdgeOut = std::make_unique<unsigned int>(0);
    auto nEdgeRawOut = std::make_unique<unsigned int>(0);

    auto trackSVNonSV = std::make_unique<std::vector<float>>();
    auto trackSVPV = std::make_unique<std::vector<float>>();
    auto trackSVSV = std::make_unique<std::vector<float>>();
    auto trackSub0 = std::make_unique<std::vector<float>>();
    auto trackSub1 = std::make_unique<std::vector<float>>();
    auto trackSub2 = std::make_unique<std::vector<float>>();

    auto edgeSrc = std::make_unique<std::vector<int>>();
    auto edgeDst = std::make_unique<std::vector<int>>();
    auto edgeProb = std::make_unique<std::vector<float>>();

    std::unique_ptr<std::vector<float>> svLogit0;
    std::unique_ptr<std::vector<float>> svLogit1;
    std::unique_ptr<std::vector<float>> svLogit2;
    std::unique_ptr<std::vector<float>> subLogit0;
    std::unique_ptr<std::vector<float>> subLogit1;
    std::unique_ptr<std::vector<float>> subLogit2;
    std::unique_ptr<std::vector<float>> edgeLogit;
    if (writeLogits_) {
      svLogit0 = std::make_unique<std::vector<float>>();
      svLogit1 = std::make_unique<std::vector<float>>();
      svLogit2 = std::make_unique<std::vector<float>>();
      subLogit0 = std::make_unique<std::vector<float>>();
      subLogit1 = std::make_unique<std::vector<float>>();
      subLogit2 = std::make_unique<std::vector<float>>();
      edgeLogit = std::make_unique<std::vector<float>>();
    }

    const auto &tracks = event.get(tracksToken_);
    const auto &lostTracks = event.get(lostTracksToken_);
    const auto &pvs = event.get(pvToken_);

    if (pvs.empty()) {
      putOutputs(event,
                 std::move(nTrackOut),
                 std::move(nEdgeOut),
                 std::move(nEdgeRawOut),
                 std::move(trackSVNonSV),
                 std::move(trackSVPV),
                 std::move(trackSVSV),
                 std::move(trackSub0),
                 std::move(trackSub1),
                 std::move(trackSub2),
                 std::move(edgeSrc),
                 std::move(edgeDst),
                 std::move(edgeProb),
                 std::move(svLogit0),
                 std::move(svLogit1),
                 std::move(svLogit2),
                 std::move(subLogit0),
                 std::move(subLogit1),
                 std::move(subLogit2),
                 std::move(edgeLogit));
      return;
    }

    const reco::Vertex &pv = pvs.front();
    const auto &ttb = setup.getData(ttbToken_);

    std::vector<reco::Track> allTracks;
    std::vector<reco::TransientTrack> transientTracks;

    auto addPacked = [&](const pat::PackedCandidateCollection &collection) {
      for (const auto &cand : collection) {
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
    *nTrackOut = static_cast<unsigned int>(nTracks);

    if (nTracks == 0) {
      putOutputs(event,
                 std::move(nTrackOut),
                 std::move(nEdgeOut),
                 std::move(nEdgeRawOut),
                 std::move(trackSVNonSV),
                 std::move(trackSVPV),
                 std::move(trackSVSV),
                 std::move(trackSub0),
                 std::move(trackSub1),
                 std::move(trackSub2),
                 std::move(edgeSrc),
                 std::move(edgeDst),
                 std::move(edgeProb),
                 std::move(svLogit0),
                 std::move(svLogit1),
                 std::move(svLogit2),
                 std::move(subLogit0),
                 std::move(subLogit1),
                 std::move(subLogit2),
                 std::move(edgeLogit));
      return;
    }

    std::vector<std::array<float, 14>> trackFeatures;
    trackFeatures.reserve(nTracks);

    for (size_t i = 0; i < nTracks; ++i) {
      const auto &trk = allTracks[i];
      const auto &tt = transientTracks[i];

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

      float dz = std::abs(trk.dz(pv.position()));
      float dzsig = (trk.dzError() > 0.f) ? std::abs(trk.dz(pv.position()) / trk.dzError()) : -999.f;

      trackFeatures.push_back({
          trk.eta(),
          trk.phi(),
          ip2d,
          ip3d,
          dz,
          dzsig,
          ip2dsig,
          ip3dsig,
          trk.p(),
          trk.pt(),
          static_cast<float>(trk.numberOfValidHits()),
          static_cast<float>(trk.hitPattern().numberOfValidPixelHits()),
          static_cast<float>(trk.hitPattern().numberOfValidStripHits()),
          static_cast<float>(trk.charge())});
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

        (*nEdgeRawOut)++;

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
        edgeFeatures.push_back({dca, deltaR, dcaSig, cpToPv, pvToPcaI, pvToPcaJ, dotI, dotJ, pairMom, invMass});
      }
    }

    const size_t nEdges = localSrc.size();
    *nEdgeOut = static_cast<unsigned int>(nEdges);

    // If there are no edges, we keep track arrays empty (conservative behavior).
    if (nEdges == 0) {
      putOutputs(event,
                 std::move(nTrackOut),
                 std::move(nEdgeOut),
                 std::move(nEdgeRawOut),
                 std::move(trackSVNonSV),
                 std::move(trackSVPV),
                 std::move(trackSVSV),
                 std::move(trackSub0),
                 std::move(trackSub1),
                 std::move(trackSub2),
                 std::move(edgeSrc),
                 std::move(edgeDst),
                 std::move(edgeProb),
                 std::move(svLogit0),
                 std::move(svLogit1),
                 std::move(svLogit2),
                 std::move(subLogit0),
                 std::move(subLogit1),
                 std::move(subLogit2),
                 std::move(edgeLogit));
      return;
    }

    std::vector<float> xInFlat;
    xInFlat.reserve(nTracks * 14);
    for (const auto &f : trackFeatures)
      xInFlat.insert(xInFlat.end(), f.begin(), f.end());

    std::vector<float> edgeIndexFlat;
    edgeIndexFlat.reserve(nEdges * 2);
    for (const auto &s : localSrc)
      edgeIndexFlat.push_back(static_cast<float>(s));
    for (const auto &d : localDst)
      edgeIndexFlat.push_back(static_cast<float>(d));

    std::vector<float> edgeAttrFlat;
    edgeAttrFlat.reserve(nEdges * 10);
    for (const auto &f : edgeFeatures)
      edgeAttrFlat.insert(edgeAttrFlat.end(), f.begin(), f.end());

    std::vector<std::string> inputNames = {"x_in", "edge_index", "edge_attr"};
    std::vector<std::vector<int64_t>> inputShapes = {
        {1, static_cast<int64_t>(nTracks), 14},
        {1, 2, static_cast<int64_t>(nEdges)},
        {1, static_cast<int64_t>(nEdges), 10},
    };
    std::vector<std::vector<float>> inputData = {xInFlat, edgeIndexFlat, edgeAttrFlat};

    const auto output = globalCache()->run(inputNames, inputData, inputShapes);
    if (output.size() < 3)
      throw cms::Exception("TrackGNNNanoProducer") << "Model output has fewer than 3 tensors.";

    const auto &svLogits = output[0];
    const auto &subLogits = output[1];
    const auto &edgeLogits = output[2];

    if (svLogits.size() != nTracks * 3 || subLogits.size() != nTracks * 3 || edgeLogits.size() != nEdges) {
      throw cms::Exception("TrackGNNNanoProducer") << "Unexpected model output sizes: "
                                                    << "sv=" << svLogits.size() << ", sub=" << subLogits.size()
                                                    << ", edge=" << edgeLogits.size() << ", N=" << nTracks
                                                    << ", E=" << nEdges;
    }

    trackSVNonSV->reserve(nTracks);
    trackSVPV->reserve(nTracks);
    trackSVSV->reserve(nTracks);
    trackSub0->reserve(nTracks);
    trackSub1->reserve(nTracks);
    trackSub2->reserve(nTracks);

    for (size_t i = 0; i < nTracks; ++i) {
      float pNode[3];
      softmax3(svLogits[i * 3], svLogits[i * 3 + 1], svLogits[i * 3 + 2], pNode);
      trackSVNonSV->push_back(pNode[0]);
      trackSVPV->push_back(pNode[1]);
      trackSVSV->push_back(pNode[2]);

      float pSub[3];
      softmax3(subLogits[i * 3], subLogits[i * 3 + 1], subLogits[i * 3 + 2], pSub);
      trackSub0->push_back(pSub[0]);
      trackSub1->push_back(pSub[1]);
      trackSub2->push_back(pSub[2]);

      if (writeLogits_) {
        svLogit0->push_back(svLogits[i * 3]);
        svLogit1->push_back(svLogits[i * 3 + 1]);
        svLogit2->push_back(svLogits[i * 3 + 2]);
        subLogit0->push_back(subLogits[i * 3]);
        subLogit1->push_back(subLogits[i * 3 + 1]);
        subLogit2->push_back(subLogits[i * 3 + 2]);
      }
    }

    edgeSrc->insert(edgeSrc->end(), localSrc.begin(), localSrc.end());
    edgeDst->insert(edgeDst->end(), localDst.begin(), localDst.end());
    edgeProb->reserve(nEdges);
    for (size_t e = 0; e < nEdges; ++e) {
      edgeProb->push_back(sigmoid(edgeLogits[e]));
      if (writeLogits_)
        edgeLogit->push_back(edgeLogits[e]);
    }

    putOutputs(event,
               std::move(nTrackOut),
               std::move(nEdgeOut),
               std::move(nEdgeRawOut),
               std::move(trackSVNonSV),
               std::move(trackSVPV),
               std::move(trackSVSV),
               std::move(trackSub0),
               std::move(trackSub1),
               std::move(trackSub2),
               std::move(edgeSrc),
               std::move(edgeDst),
               std::move(edgeProb),
               std::move(svLogit0),
               std::move(svLogit1),
               std::move(svLogit2),
               std::move(subLogit0),
               std::move(subLogit1),
               std::move(subLogit2),
               std::move(edgeLogit));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("tracks", edm::InputTag("packedPFCandidates"));
    desc.add<edm::InputTag>("losttracks", edm::InputTag("lostTracks", "", "PAT"));
    desc.add<edm::InputTag>("primaryVertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
    desc.add<edm::FileInPath>("model_path", edm::FileInPath("dispV/dispVAnalyzer/data/bhive_hcmod_1703.onnx"));

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

    descriptions.add("trackGNNNano", desc);
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

  void putOutputs(edm::Event &event,
                  std::unique_ptr<unsigned int> nTrackOut,
                  std::unique_ptr<unsigned int> nEdgeOut,
                  std::unique_ptr<unsigned int> nEdgeRawOut,
                  std::unique_ptr<std::vector<float>> trackSVNonSV,
                  std::unique_ptr<std::vector<float>> trackSVPV,
                  std::unique_ptr<std::vector<float>> trackSVSV,
                  std::unique_ptr<std::vector<float>> trackSub0,
                  std::unique_ptr<std::vector<float>> trackSub1,
                  std::unique_ptr<std::vector<float>> trackSub2,
                  std::unique_ptr<std::vector<int>> edgeSrc,
                  std::unique_ptr<std::vector<int>> edgeDst,
                  std::unique_ptr<std::vector<float>> edgeProb,
                  std::unique_ptr<std::vector<float>> svLogit0,
                  std::unique_ptr<std::vector<float>> svLogit1,
                  std::unique_ptr<std::vector<float>> svLogit2,
                  std::unique_ptr<std::vector<float>> subLogit0,
                  std::unique_ptr<std::vector<float>> subLogit1,
                  std::unique_ptr<std::vector<float>> subLogit2,
                  std::unique_ptr<std::vector<float>> edgeLogit) const {
    event.put(std::move(nTrackOut), "nTrackGNNTrack");
    event.put(std::move(nEdgeOut), "nTrackGNNEdge");
    event.put(std::move(nEdgeRawOut), "nTrackGNNEdgeRaw");

    event.put(std::move(trackSVNonSV), "TrackGNN_track_SV_prob_nonSV");
    event.put(std::move(trackSVPV), "TrackGNN_track_SV_prob_PV");
    event.put(std::move(trackSVSV), "TrackGNN_track_SV_prob_SV");

    event.put(std::move(trackSub0), "TrackGNN_track_sub_prob_0");
    event.put(std::move(trackSub1), "TrackGNN_track_sub_prob_1");
    event.put(std::move(trackSub2), "TrackGNN_track_sub_prob_2");

    event.put(std::move(edgeSrc), "TrackGNN_edge_src");
    event.put(std::move(edgeDst), "TrackGNN_edge_dst");
    event.put(std::move(edgeProb), "TrackGNN_edge_prob");

    if (writeLogits_) {
      event.put(std::move(svLogit0), "TrackGNN_track_SV_logit_nonSV");
      event.put(std::move(svLogit1), "TrackGNN_track_SV_logit_PV");
      event.put(std::move(svLogit2), "TrackGNN_track_SV_logit_SV");
      event.put(std::move(subLogit0), "TrackGNN_track_sub_logit_0");
      event.put(std::move(subLogit1), "TrackGNN_track_sub_logit_1");
      event.put(std::move(subLogit2), "TrackGNN_track_sub_logit_2");
      event.put(std::move(edgeLogit), "TrackGNN_edge_logit");
    }
  }

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
