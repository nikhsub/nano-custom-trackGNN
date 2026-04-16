// -*- C++ -*-
//
// Package:    Demo/DemoAnalyzer
// Class:      DemoAnalyzer
//
/**\class DemoAnalyzer DemoAnalyzer.cc Demo/DemoAnalyzer/plugins/DemoAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Nikhilesh Venkatasubramanian
//         Created:  Tue, 16 Apr 2024 18:10:25 GMT
//
//

//
// constructors and destructor

#include "dispV/dispVAnalyzer/interface/DemoAnalyzer.h"
#include "dispV/dispVAnalyzer/interface/matchedHadronsToSV.h"
#include <iostream>
#include <omp.h>
#include <unordered_set>
#include <iomanip>
#include <unordered_map>
#include "TH1F.h"

// function to get 3D distance bewtween all SV and all GV
static std::vector<std::vector<float>> computeDistanceMatrix(
                const std::vector<float>& SV_x,
                const std::vector<float>& SV_y,
                const std::vector<float>& SV_z,
                const std::vector<float>& Hadron_GVx,
                const std::vector<float>& Hadron_GVy,
                const std::vector<float>& Hadron_GVz) {
    // computeDistanceMatrix
    // Returns
    // distances = Matrix of Euclidean distances between SV and GV
    
    size_t nSV = SV_x.size();
    size_t nHadron = Hadron_GVx.size();
    
    // 2D vector initialized to 999 nSV x nHadron
    std::vector<std::vector<float>> distances(nSV, std::vector<float>(nHadron, 999.0));

    for (size_t i = 0; i < nSV; ++i) {
        for (size_t j = 0; j < nHadron; ++j) {
            float dx = SV_x[i] - Hadron_GVx[j];
            float dy = SV_y[i] - Hadron_GVy[j];
            float dz = SV_z[i] - Hadron_GVz[j];
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            distances[i][j] = dist;
        }
    }

    return distances;
}

// Constructor
DemoAnalyzer::DemoAnalyzer(const edm::ParameterSet& iConfig, const ONNXRuntime *cache):
    // Member Initialization
    //esConsumes -> token to EventSetup module
    //consumes -> token to Event data module
	theTTBToken(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
	training_ (iConfig.getUntrackedParameter<bool>("training")),
	TrackCollT_ (consumes<pat::PackedCandidateCollection>(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))),
	PVCollT_ (consumes<reco::VertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("primaryVertices"))),
	SVCollT_ (consumes<edm::View<reco::VertexCompositePtrCandidate>>(iConfig.getUntrackedParameter<edm::InputTag>("secVertices"))),
  	LostTrackCollT_ (consumes<pat::PackedCandidateCollection>(iConfig.getUntrackedParameter<edm::InputTag>("losttracks"))),
	jet_collT_ (consumes<edm::View<reco::Jet> >(iConfig.getUntrackedParameter<edm::InputTag>("jets"))),
        beamspotToken_ (consumes<reco::BeamSpot>(iConfig.getUntrackedParameter<edm::InputTag>("beamspot"))),
	prunedGenToken_(consumes<edm::View<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("pruned"))),
  	packedGenToken_(consumes<edm::View<pat::PackedGenParticle> >(iConfig.getParameter<edm::InputTag>("packed"))),
	mergedGenToken_(consumes<edm::View<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("merged"))),
	TrackPtCut_(iConfig.getUntrackedParameter<double>("TrackPtCut")),
	TrackPredCut_(iConfig.getUntrackedParameter<double>("TrackPredCut")),
	vtxconfig_(iConfig.getUntrackedParameter<edm::ParameterSet>("vertexfitter")),
    	vtxmaker_(vtxconfig_),
	PupInfoT_ (consumes<std::vector<PileupSummaryInfo>>(iConfig.getUntrackedParameter<edm::InputTag>("addPileupInfo"))),
	vtxweight_(iConfig.getUntrackedParameter<double>("vtxweight")),
	clusterizer(new TracksClusteringFromDisplacedSeed(iConfig.getParameter<edm::ParameterSet>("clusterizer"))),
	genmatch_csv_(iConfig.getParameter<edm::FileInPath>("genmatch_csv").fullPath())
{
	edm::Service<TFileService> fs;	
   	tree = fs->make<TTree>("tree", "tree");
}

// Destructor
DemoAnalyzer::~DemoAnalyzer() {}


// Tell ONNXRunTime where is file location
std::unique_ptr<ONNXRuntime> DemoAnalyzer::initializeGlobalCache(const edm::ParameterSet &iConfig) 
{
    return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void DemoAnalyzer::globalEndJob(const ONNXRuntime *cache) {}


int DemoAnalyzer::checkPDG(int abs_pdg)
// Returns:
// 1 B hadron
// 2 D hadron
// 3 S hadron
// 4 Tau
// 5 anything else
{
    static const std::unordered_set<int> pdgSet_B = {
        521, 511, 531, 541,        // Bottom mesons
        5122,                      // Lambda_b0
        5132, 5232, 5332,          // Xi_b-, Xi_b0, Omega_b-
        5142, 5242, 5342,          // Xi_bc0, Xi_bc+, Omega_bc0
        5512, 5532, 5542, 5554     // Xi_bb-, Omega_bb-, Omega_bbc0, Omega_bbb-
    };

    static const std::unordered_set<int> pdgSet_D = {
        411, 421, 431,             // Charmed mesons
        4122,                      // Lambda_c
        4232, 4132, 4332,          // Xi_c+, Xi’c0, Omega_c0
        4412, 4422, 4432, 4444     // Xi_cc+, Xi_cc++, Omega_cc+, Omega_ccc++
    };

    static const std::unordered_set<int> pdgSet_S = {
        3122, 3222, 3212,          // Lambda, Sigma+, Sigma0
        3312, 3322, 3334           // Xi-, Xi0, Omega-
    };

    static const std::unordered_set<int> pdgSet_Tau = {15};

    if (pdgSet_B.count(abs_pdg))      return 1;
    if (pdgSet_D.count(abs_pdg))      return 2;
    if (pdgSet_S.count(abs_pdg))      return 3;
    if (pdgSet_Tau.count(abs_pdg))    return 4;
    return 5;
}

int DemoAnalyzer::getDaughterLabel(const reco::GenParticle* dau)
{
    // Label meaning:
    // 0 = Primary (hard scatter)
    // 1 = Pileup
    // 2 = fromB
    // 3 = fromBC
    // 4 = fromC
    // 5 = OtherSecondary (S, τ, conversions)

    static const std::unordered_set<int> pdgSet_B = {
        521, 511, 531, 541, 5122, 5132, 5232, 5332,
        5142, 5242, 5342, 5512, 5532, 5542, 5554
    };
    static const std::unordered_set<int> pdgSet_C = {
        411, 421, 431, 4122, 4232, 4132, 4332,
        4412, 4422, 4432, 4444
    };
    static const std::unordered_set<int> pdgSet_S = {
        3122, 3222, 3212, 3312, 3322, 3334
    };
    static const std::unordered_set<int> pdgSet_Tau = {15};

    // --- 1. Collision ID: cleanest PU separation ---
    if (dau->collisionId() > 0)
        return 1; // pileup (non-primary collision)

    // --- 2. Initialize ancestry flags ---
    bool foundB   = false;
    bool foundC   = false;

    // --- 3. Traverse ancestry chain ---
    const reco::Candidate* mom = dau->mother(0);
    int guard = 0;
    while (mom && guard++ < 100) {
        const int apdg = std::abs(mom->pdgId());

        if (pdgSet_B.count(apdg))   foundB   = true;
        if (pdgSet_C.count(apdg))   foundC   = true;

        // stop once we've reached a B hadron (no need to go higher)
        if (foundB) break;

        mom = mom->mother(0);
    }

    // --- 4. Heavy-flavor classification ---
    if (foundB && foundC) return 3; // fromBC
    if (foundB)           return 2; // fromB
    if (foundC)           return 4; // fromC

    // --- 5. Other displaced / secondary sources ---
    const reco::Candidate* mother = dau->mother(0);
    if (mother) {
        int mabs = std::abs(mother->pdgId());
        if (pdgSet_S.count(mabs) || pdgSet_Tau.count(mabs) || mabs == 22)
            return 5;  // strange hadron, τ, or photon conversion
    }

    // --- 6. Otherwise: primary hard scatter ---
    return 0;
}



bool DemoAnalyzer::isGoodVtx(TransientVertex& tVTX){

   reco::Vertex tmpvtx(tVTX);
   //return (tVTX.isValid() &&
   // !tmpvtx.isFake() &&
   // (tmpvtx.nTracks(vtxweight_)>1) &&
   // (tmpvtx.normalizedChi2()>0) &&
   // (tmpvtx.normalizedChi2()<10));
   return tVTX.isValid();
}



std::vector<TransientVertex> DemoAnalyzer::TrackVertexRefit(std::vector<reco::TransientTrack> &Tracks,
                                                            std::vector<TransientVertex> &VTXs)
    //TrackVertexRefit
    // INPUTS
    // - tracks
    // - inputs
    // OUTPUTS
    // - vertex (new collection based on tracks and inputs used as inputs)
{
    AdaptiveVertexFitter theAVF(GeometricAnnealing(3, 256, 0.25),
				                DefaultLinearizationPointFinder(),
                                KalmanVertexUpdator<5>(),
                                KalmanVertexTrackCompatibilityEstimator<5>(),
                                KalmanVertexSmoother());
 
  std::vector<TransientVertex> newVTXs;

  for(std::vector<TransientVertex>::const_iterator sv = VTXs.begin(); sv != VTXs.end(); ++sv){
      GlobalPoint ssv = sv->position();
      reco::Vertex tmpvtx = reco::Vertex(*sv);
      std::vector<reco::TransientTrack> selTrks; 
      for(std::vector<reco::TransientTrack>::const_iterator trk = Tracks.begin(); trk!=Tracks.end(); ++trk){
	  if (trk->track().pt() <= 0 || !trk->track().charge()) continue;
          Measurement1D ip3d = IPTools::absoluteImpactParameter3D(*trk, tmpvtx).second;
          if(ip3d.significance()<5.0 || sv->trackWeight(*trk)>0.5){
              selTrks.push_back(*trk);
          }
      }
      
      if(selTrks.size()>=2){
          TransientVertex newsv = theAVF.vertex(selTrks, ssv);
          if(isGoodVtx(newsv))  newVTXs.push_back(newsv);
      }
  
  }
  return newVTXs;
}

void DemoAnalyzer::vertexMerge(std::vector<TransientVertex>& VTXs, double maxFraction, double minSignificance) 
//merges (removes) close, overlapping vertices based on shared tracks and distance significance.
// INPUTS
// - VTXs : vertex collection
// - maxFraction: max allowed fraction of shared tracks before merging
// - minSignificance: max allowed significance of vertex distance before merging
{
	
    if (VTXs.empty()) return;
    VertexDistance3D dist;
	
   for (auto sv = VTXs.begin(); sv != VTXs.end(); /* no ++ here */) {
	if (!sv->isValid()) {
        sv = VTXs.erase(sv);  // Remove invalid vertex right away
        continue;
        }
        bool shared = false;
        VertexState s1 = sv->vertexState();
        const auto& tracks_i = sv->originalTracks();

        for (auto sv2 = VTXs.begin(); sv2 != VTXs.end(); ++sv2) {
            if (sv == sv2 || !sv2->isValid()) continue;

            VertexState s2 = sv2->vertexState();
            const auto& tracks_j = sv2->originalTracks();

            // Count shared tracks by comparing pt values
            int sharedTracks = 0;
            for (const auto& ti : tracks_i) {
                for (const auto& tj : tracks_j) {
                    double dpt = std::abs(ti.track().pt() - tj.track().pt());
                    if (dpt < 1e-3) {
                        ++sharedTracks;
                        break;  // each ti only counts once
                    }
                }
            }

            double sharedFrac = static_cast<double>(sharedTracks) / std::min(tracks_i.size(), tracks_j.size());
            double sig = dist.distance(s1, s2).significance();

            if (sharedFrac > maxFraction && sig < minSignificance) {
                shared = true;
                break;
            }
        }

        if (shared) {
            sv = VTXs.erase(sv);  // erase returns next valid iterator
        } else {
            ++sv;
        }
    } 
}

std::vector<TransientVertex>
DemoAnalyzer::TrackVertexArbitrator(const reco::Vertex& pv,
                                    const edm::Handle<reco::BeamSpot>& bsH,
                                    const std::vector<TransientVertex>& seedSVs,
                                    std::vector<reco::TransientTrack>& allTTs,
                                    // arbitration cuts
                                    double dRCut,
                                    double distCut,
                                    double sigCut,
                                    double dLenFraction,
                                    double fitterSigmacut,
                                    double fitterTini,
                                    double fitterRatio,
                                    double maxTimeSig,
                                    // track quality cuts
                                    int trackMinLayers,
                                    double trackMinPt,
                                    int trackMinPixels) {
  std::vector<TransientVertex> out;
  VertexDistance3D vdist;
  GlobalPoint ppv(pv.position().x(), pv.position().y(), pv.position().z());

  AdaptiveVertexFitter avf(
      GeometricAnnealing(fitterSigmacut, fitterTini, fitterRatio),
      DefaultLinearizationPointFinder(),
      KalmanVertexUpdator<5>(),
      KalmanVertexTrackCompatibilityEstimator<5>(),
      KalmanVertexSmoother());

  // handle signed dR cut like CMSSW
  double sign = 1.0;
  if (dRCut < 0) {
    sign = -1.0;
  }
  double dR2cut = dRCut * dRCut;

  std::unordered_map<size_t, Measurement1D> cachedIP;

  for (auto const& sv : seedSVs) {

   if (!sv.isValid()) continue;

    // time info from SV
    const double svTime = sv.time();
    const double svTimeCov = sv.positionError().czz();  // cov(3,3)
    const bool svHasTime = (svTimeCov > 0.);

    GlobalPoint ssv(sv.position().x(), sv.position().y(), sv.position().z());
    GlobalVector flightDir = ssv - ppv;

    Measurement1D dlen = vdist.distance(
        pv,
        VertexState(GlobalPoint(sv.position().x(),
                                sv.position().y(),
                                sv.position().z()),
                    sv.positionError()));

    const auto& svTracks = sv.originalTracks();
    std::vector<reco::TransientTrack> selTracks;
    selTracks.reserve(svTracks.size());

    for (size_t i = 0; i < allTTs.size(); ++i) {
      reco::TransientTrack& tt = allTTs[i];
      if (!tt.isValid()) continue;

      // === Track quality cuts (like CMSSW::trackFilterArbitrator) ===
      if (tt.track().hitPattern().trackerLayersWithMeasurement() < trackMinLayers) continue;
      if (tt.track().pt() < trackMinPt) continue;
      if (tt.track().hitPattern().numberOfValidPixelHits() < trackMinPixels) continue;

      tt.setBeamSpot(*bsH);

      // === Track membership weight (use CMSSW-style) ===
      //float weight = 0.;
      //for (const auto& t0 : svTracks) {
      //  double dpt  = std::abs(tt.track().pt()  - t0.track().pt());
      //  double deta = std::abs(tt.track().eta() - t0.track().eta());
      //  double dphi = std::abs(reco::deltaPhi(tt.track().phi(), t0.track().phi()));
      //  if (dpt < 1e-3 && deta < 1e-3 && dphi < 1e-3) {
      //    weight = 1.0;
      //    break;
      //  }
      //}
      float weight = sv.trackWeight(tt);

      // === IP significance wrt PV (cached) ===
      Measurement1D ipv;
      if (cachedIP.count(i)) {
        ipv = cachedIP[i];
      } else {
        auto ipvp = IPTools::absoluteImpactParameter3D(tt, pv);
        cachedIP[i] = ipvp.second;
        ipv = ipvp.second;
      }

      // === Extrapolate track to SV position ===
      AnalyticalImpactPointExtrapolator extrap(tt.field());
      TrajectoryStateOnSurface tsos =
          extrap.extrapolate(tt.impactPointState(), ssv);

      if (!tsos.isValid()) continue;

      GlobalPoint refPoint = tsos.globalPosition();
      GlobalError refErr = tsos.cartesianError().position();

      Measurement1D isv = vdist.distance(
          VertexState(ssv, sv.positionError()),
          VertexState(refPoint, refErr));

      // === ΔR with signed flight direction ===
      float dR2 = Geom::deltaR2(((sign > 0) ? flightDir : -flightDir), tt.track());

      // === Time significance ===
      double timeSig = 0.;
      if (svHasTime && edm::isFinite(tt.timeExt())) {
        double tErr = std::sqrt(std::pow(tt.dtErrorExt(), 2) + svTimeCov);
        timeSig = std::abs(tt.timeExt() - svTime) / tErr;
      }

      // === Arbitration decision ===
      if (weight > 0. ||
          (isv.significance() < sigCut &&
           isv.value() < distCut &&
           isv.value() < dlen.value() * dLenFraction &&
           timeSig < maxTimeSig)) {

        if ((isv.value() < ipv.value()) &&
            isv.value() < distCut &&
            isv.value() < dlen.value() * dLenFraction &&
            dR2 < dR2cut &&
            timeSig < maxTimeSig) {
          selTracks.push_back(tt);
        } else {
          // Extra acceptance for member tracks, like CMSSW
          if (weight > 0.5 &&
              isv.value() <= ipv.value() &&
              dR2 < dR2cut &&
              timeSig < maxTimeSig) {
            selTracks.push_back(tt);
          }
        }
      }
    }

    // === Vertex refit ===
    if (selTracks.size() >= 2) {
      TransientVertex tv = avf.vertex(selTracks, ssv);
      if (tv.isValid()) {
        // Update vertex time like CMSSW if PV covariance is valid
        if (pv.covariance(3, 3) > 0.) {
          svhelper::updateVertexTime(tv);
        }
        out.push_back(std::move(tv));
      }
    }
  }

  return out;
}

inline float DemoAnalyzer::sigmoid(float x) {
    if (std::isnan(x)) return 0.0f;  // handle NaN safely
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = std::exp(x);
        return z / (1.0f + z);
    }
}






// ------------ method called for each event  ------------
void DemoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace reco;
  using namespace pat;

  run_ = iEvent.id().run();
  lumi_ = iEvent.luminosityBlock();
  evt_ = iEvent.id().event();

  const SigMatchEntry* genmatch = nullptr;

  auto key = std::make_tuple(run_, lumi_, evt_);
  auto it = sigMatchMap_.find(key);
  if (it != sigMatchMap_.end()) {
      genmatch = &it->second;
  }

  std::unordered_map<int, std::pair<int,int>> truthByTrack; // trkIdx -> (label, hadidx)
  truthByTrack.reserve(256);

  std::unordered_set<int> matchedTruthTracks;
  matchedTruthTracks.reserve(256);
  
  if (genmatch) {
    for (size_t k = 0; k < genmatch->indices.size(); ++k) {
      int trk = genmatch->indices[k];
      int lab = genmatch->labels[k];
      int hid = genmatch->hadidx[k];
      truthByTrack.emplace(trk, std::make_pair(lab, hid));
    }
  }

  nPU = 0;
    //vectors defined in .h
  Hadron_pt.clear();
  Hadron_SVIdx.clear();
  Hadron_SVDistance.clear();
  Hadron_eta.clear();
  Hadron_phi.clear();
  Hadron_GVx.clear();
  Hadron_GVy.clear();
  Hadron_GVz.clear(); 
  nHadrons.clear();
  nGV.clear();
  nGV_Tau.clear();
  nGV_B.clear();
  nGV_S.clear();
  nGV_D.clear();
  GV_flag.clear();
  nDaughters.clear();
  nDaughters_B.clear();
  nDaughters_D.clear();
  Daughters_hadidx.clear();
  Daughters_flav.clear();
  Daughters_label.clear();
  Daughters_pt.clear();
  //Daughters_pdg.clear();
  Daughters_eta.clear();
  Daughters_phi.clear();
  Daughters_charge.clear();

  ntrks.clear();
  trk_ip2d.clear();
  trk_ip3d.clear();
  trk_ipz.clear();
  trk_ipzsig.clear();
  trk_ip2dsig.clear();
  trk_ip3dsig.clear();
  trk_p.clear();
  trk_pt.clear();
  trk_eta.clear();
  trk_phi.clear();
  trk_charge.clear();
  trk_nValid.clear();
  trk_nValidPixel.clear();
  trk_nValidStrip.clear();

  //njets.clear();
  //jet_pt.clear();
  //jet_eta.clear();
  //jet_phi.clear();
  trk_i.clear();
  trk_j.clear();
  deltaR.clear();
  dca.clear();
  dca_sig.clear();
  cptopv.clear();
  pvtoPCA_i.clear();
  pvtoPCA_j.clear();
  dotprod_i.clear();
  dotprod_j.clear();
  pair_mom.clear();
  pair_invmass.clear();

  nSVs.clear();
  SV_x.clear();
  SV_y.clear();
  SV_z.clear();
  SV_pt.clear();
  SV_mass.clear();
  SV_ntrks.clear();
  SVtrk_pt.clear();
  SVtrk_SVIdx.clear();
  SVtrk_eta.clear();
  SVtrk_phi.clear();
  SVtrk_ipz.clear();
  SVtrk_ipzsig.clear();
  SVtrk_ipxy.clear();
  SVtrk_ipxysig.clear();
  SVtrk_ip3d.clear();
  SVtrk_ip3dsig.clear();

  preds.clear();
  cut.clear();

  nSVs_reco.clear();
  SV_x_reco.clear();
  SVrecoTrk_SVrecoIdx.clear();
  SVrecoTrk_pt.clear();
  SVrecoTrk_eta.clear();
  SVrecoTrk_phi.clear();
  SVrecoTrk_ip3d.clear();
  SVrecoTrk_ip2d.clear();
  SV_y_reco.clear();
  SV_z_reco.clear();
  SV_reco_nTracks.clear();
  SV_chi2_reco.clear();
  Hadron_SVRecoIdx.clear();
  Hadron_SVRecoDistance.clear();

  sv_score_sig.clear();
  sv_score_bkg.clear();
  edge_score_sig.clear();
  edge_score_bkg.clear();

  nHad_B.clear();
  nHad_BtoC.clear();
  nHad_C.clear();
  eff_B.clear();
  eff_BtoC.clear();
  eff_C.clear();
  fake_B.clear();
  fake_BtoC.clear();
  fake_C.clear();


  Handle<PackedCandidateCollection> patcan;
  Handle<PackedCandidateCollection> losttracks;
  Handle<edm::View<reco::Jet> > jet_coll;
  Handle<edm::View<reco::GenParticle> > pruned;
  Handle<reco::BeamSpot> beamSpotHandle;
  Handle<edm::View<pat::PackedGenParticle> > packed;
  Handle<edm::View<reco::GenParticle> > merged;
  Handle<reco::VertexCollection> pvHandle;
  Handle<edm::View<reco::VertexCompositePtrCandidate>> svHandle ;
  Handle<std::vector< PileupSummaryInfo > > PupInfo;

  std::vector<reco::Track> alltracks;
  std::vector<reco::Track> alltracks_ForArbitration;

  iEvent.getByToken(TrackCollT_, patcan);
  iEvent.getByToken(LostTrackCollT_, losttracks);
  iEvent.getByToken(jet_collT_, jet_coll);
  iEvent.getByToken(prunedGenToken_,pruned);
  iEvent.getByToken(packedGenToken_,packed);
  iEvent.getByToken(mergedGenToken_, merged);
  iEvent.getByToken(PVCollT_, pvHandle);
  iEvent.getByToken(SVCollT_, svHandle);
  iEvent.getByToken(beamspotToken_, beamSpotHandle);

  //std::cout<<"Merged size:"<<merged->size()<<std::endl;
  //std::cout<<"Packed size:"<<packed->size()<<std::endl;
  //std::cout<<"Pruned size:"<<pruned->size()<<std::endl;

  const auto& theB = &iSetup.getData(theTTBToken);
  reco::Vertex pv = (*pvHandle)[0];



  GlobalVector direction(1,0,0);
  direction = direction.unit();

  iEvent.getByToken(PupInfoT_, PupInfo);
      std::vector<PileupSummaryInfo>::const_iterator PVI;
       for(PVI = PupInfo->begin(); PVI != PupInfo->end(); ++PVI){

           int BX = PVI->getBunchCrossing();
           if(BX ==0) {
               nPU = PVI->getTrueNumInteractions();
               continue;
           }

   }


  // From PFCandidates to tracks
  //Fil two arrays of tracks
  // alltracks
  // alltracjs for arbitration
  for (auto const& itrack : *patcan){
       if (itrack.trackHighPurity() && itrack.hasTrackDetails()){
           reco::Track tmptrk = itrack.pseudoTrack();
           if (tmptrk.quality(reco::TrackBase::highPurity) && tmptrk.pt()> 0.8 && tmptrk.charge()!=0){
            alltracks_ForArbitration.push_back(tmptrk);
           }
           if (tmptrk.quality(reco::TrackBase::highPurity) && tmptrk.pt()> TrackPtCut_ && tmptrk.charge()!=0){
	       alltracks.push_back(tmptrk);
           }
       }
   }

   for (auto const& itrack : *losttracks){
       if (itrack.trackHighPurity() && itrack.hasTrackDetails()){
           reco::Track tmptrk = itrack.pseudoTrack();
        if (tmptrk.quality(reco::TrackBase::highPurity) && tmptrk.pt()> 0.8 && tmptrk.charge()!=0){
            alltracks_ForArbitration.push_back(tmptrk);
           }
        if (tmptrk.quality(reco::TrackBase::highPurity) && tmptrk.pt()> TrackPtCut_ && tmptrk.charge()!=0){
               alltracks.push_back(itrack.pseudoTrack());
           }
       }
   }
    //const reco::BeamSpot& beamSpot = *beamSpotHandle;  
   //int njet = 0;
   //for (auto const& ijet: *jet_coll){
   //	jet_pt.push_back(ijet.pt());
   //     jet_eta.push_back(ijet.eta());
   //     jet_phi.push_back(ijet.phi());
   //     njet++;
   //}
   //njets.push_back(njet);
   
   int ntrk = 0;
   size_t num_tracks = alltracks.size();
   std::vector<reco::TransientTrack> t_trks(num_tracks);
   std::vector<Measurement1D> ip2d_vals(num_tracks);
   std::vector<Measurement1D> ip3d_vals(num_tracks);

   std::vector<int> origToNode(num_tracks, -1);
   std::vector<int> nodeToOrig;
   nodeToOrig.reserve(num_tracks);
   int node = 0;

   for (size_t i = 0; i< num_tracks; ++i) {
        t_trks[i] = (*theB).build(alltracks[i]);
        if (!(t_trks[i].isValid())) continue;

	origToNode[i] = node;
 	nodeToOrig.push_back((int)i);	
 
        ip2d_vals[i] = IPTools::signedTransverseImpactParameter(t_trks[i], direction, pv).second;
        ip3d_vals[i] = IPTools::signedImpactParameter3D(t_trks[i], direction, pv).second;

	double ip_z = t_trks[i].track().dz(pv.position());
        double ip_z_sig = ip_z / t_trks[i].track().dzError();

        trk_ipz.push_back(ip_z);
        trk_ipzsig.push_back(ip_z_sig);
        trk_ip2d.push_back(std::abs(ip2d_vals[i].value()));
        trk_ip3d.push_back(std::abs(ip3d_vals[i].value()));
        trk_ip2dsig.push_back(std::abs(ip2d_vals[i].significance()));
        trk_ip3dsig.push_back(std::abs(ip3d_vals[i].significance()));
        trk_p.push_back(alltracks[i].p());
        trk_pt.push_back(alltracks[i].pt());
        trk_eta.push_back(alltracks[i].eta());
        trk_phi.push_back(alltracks[i].phi());
        trk_charge.push_back(alltracks[i].charge());
        trk_nValid.push_back(alltracks[i].numberOfValidHits());
        trk_nValidPixel.push_back(alltracks[i].hitPattern().numberOfValidPixelHits());
        trk_nValidStrip.push_back(alltracks[i].hitPattern().numberOfValidStripHits());
        ntrk++;

	node++;
    }
    
    size_t estimated_pairs = num_tracks * num_tracks / 2;  // Approximate number of pairs
    trk_i.reserve(estimated_pairs);
    trk_j.reserve(estimated_pairs);
    deltaR.reserve(estimated_pairs);
    dca.reserve(estimated_pairs);
    dca_sig.reserve(estimated_pairs);
    cptopv.reserve(estimated_pairs);
    pvtoPCA_i.reserve(estimated_pairs);
    pvtoPCA_j.reserve(estimated_pairs);
    dotprod_i.reserve(estimated_pairs);
    dotprod_j.reserve(estimated_pairs);
    pair_mom.reserve(estimated_pairs);
    pair_invmass.reserve(estimated_pairs); 
	
   // Parallelize the outer loop
    #pragma omp parallel for
    for (size_t i = 0; i < num_tracks; ++i) {
        if (!t_trks[i].isValid()) continue;
    
        for (size_t j = i + 1; j < num_tracks; ++j) {
            if (!t_trks[j].isValid()) continue;
    
            float eta1 = alltracks[i].eta();
            float phi1 = alltracks[i].phi();
            float eta2 = alltracks[j].eta();
            float phi2 = alltracks[j].phi();
            float delta_r_val = reco::deltaR(eta1, phi1, eta2, phi2);
    
            if (delta_r_val < 2e-4 or delta_r_val > 1) continue;
	
	    const float PION_MASS = 0.13957018; // GeV (charged pion mass)
        
            // Get 3-momenta (px, py, pz)
            float px1 = alltracks[i].px();
            float py1 = alltracks[i].py();
            float pz1 = alltracks[i].pz();
            
            float px2 = alltracks[j].px();
            float py2 = alltracks[j].py();
            float pz2 = alltracks[j].pz();

            // Calculate energies (E = sqrt(p² + m²))
            float e1 = sqrt(px1*px1 + py1*py1 + pz1*pz1 + PION_MASS*PION_MASS);
            float e2 = sqrt(px2*px2 + py2*py2 + pz2*pz2 + PION_MASS*PION_MASS);

            // Compute invariant mass
            float sum_px = px1 + px2;
            float sum_py = py1 + py2;
            float sum_pz = pz1 + pz2;
            float sum_e  = e1 + e2;
            
            float inv_mass = sqrt(sum_e*sum_e - (sum_px*sum_px + sum_py*sum_py + sum_pz*sum_pz));

	    if(inv_mass > 20.0) continue;
    
            float dca_val = -1.0;  // Default invalid value
	    float cptopv_val = -1.0;
            float pvToPCAseed_val = -1.0;
            float pvToPCAtrack_val = -1.0;
            float dotprodTrack_val = -999.0;
            float dotprodSeed_val = -999.0;
            float dcaSig_val = -1.0;
            float pairMomentumMag = -1.0;

	    TwoTrackMinimumDistance minDist;
            if (minDist.calculate(t_trks[i].impactPointState(), t_trks[j].impactPointState())) {

            	VertexDistance3D distanceComputer;
            	auto m = distanceComputer.distance(
            	VertexState(minDist.points().second, t_trks[i].impactPointState().cartesianError().position()),
            	VertexState(minDist.points().first, t_trks[j].impactPointState().cartesianError().position()));
            	dca_val = m.value();
	    	if(m.error() > 0){
	    	    dcaSig_val = m.value() / m.error();
	    	}
	
	    	GlobalPoint cp(minDist.crossingPoint());
                GlobalPoint pvp(pv.x(), pv.y(), pv.z());
   	    	 
	    	GlobalPoint seedPCA = minDist.points().second;  // PCA of track i (seed)
	    	GlobalPoint trackPCA = minDist.points().first;  // PCA of track j
   	    	
	    	pvToPCAseed_val = (seedPCA - pvp).mag();      // Distance PV to seed track's PCA
	    	pvToPCAtrack_val = (trackPCA - pvp).mag();    // Distance PV to other track's PCA
            	
            	// Calculate additional variables
            	cptopv_val = (cp - pvp).mag();
            	dotprodTrack_val = (trackPCA - pvp).unit().dot(t_trks[j].impactPointState().globalDirection().unit());
            	dotprodSeed_val = (seedPCA - pvp).unit().dot(t_trks[i].impactPointState().globalDirection().unit());
            	
            	// Pair momentum
            	GlobalVector pairMomentum((Basic3DVector<float>)(t_trks[i].track().momentum() + t_trks[j].track().momentum()));
            	pairMomentumMag = pairMomentum.mag();
	     }

	     if (dca_val < 1e-8 or dca_val > 1 or dcaSig_val > 100 or cptopv_val < 4e-4 or cptopv_val > 20 or pvToPCAseed_val > 20 or pvToPCAtrack_val > 20) continue;	
	     if (pairMomentumMag < 0.05 or pairMomentumMag > 100) continue;
    
            #pragma omp critical
            {
                trk_i.push_back(i);
                trk_j.push_back(j);
                deltaR.push_back(delta_r_val);
                dca.push_back(dca_val);
		dca_sig.push_back(dcaSig_val);
		cptopv.push_back(cptopv_val);
		pvtoPCA_j.push_back(pvToPCAtrack_val);
		pvtoPCA_i.push_back(pvToPCAseed_val);
		dotprod_j.push_back(dotprodTrack_val);
		dotprod_i.push_back(dotprodSeed_val);
		pair_mom.push_back(pairMomentumMag);
		pair_invmass.push_back(inv_mass);
            }
        }
    }

   ntrks.push_back(ntrk);

  
   if(!training_)
   {
   	std::vector<std::vector<float>> track_features;
   	std::vector<std::vector<float>> edge_features;
   	std::vector<int64_t> edge_i, edge_j;
	
	   // Step 1: Build track_features, and track bad ones
   	for (size_t i = 0; i < static_cast<size_t>(ntrk); ++i) {
   	    std::vector<float> features = {
   	        trk_eta[i],
   	        trk_phi[i],
   	        trk_ip2d[i],
   	        trk_ip3d[i],
   	        trk_ipz[i],
   	        trk_ipzsig[i],
   	        trk_ip2dsig[i],
   	        trk_ip3dsig[i],
   	        trk_p[i],
   	        trk_pt[i],
   	        static_cast<float>(trk_nValid[i]),
   	        static_cast<float>(trk_nValidPixel[i]),
   	        static_cast<float>(trk_nValidStrip[i]),
   	        static_cast<float>(trk_charge[i])
   	    };
   	
   	    bool has_nan = false;
   	    for (float val : features) {
   	        if (!std::isfinite(val)) {
   	            has_nan = true;
   	            break;
   	        }
   	    }
   	
   	    if (has_nan) {
   	        features = {
   	         -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f, -999.0f,  // 10 dummy float features
   	         -1.0f, -1.0f, -1.0f,  // 3 dummy int features as float
   	         -3.0f          // charge dummy
   	        };
   	    }
   	
   	    track_features.push_back(features);
   	}

   	 
   	
   	for (size_t idx = 0; idx < trk_i.size(); ++idx) {
   	    int oi = trk_i[idx];
   	    int oj = trk_j[idx];

	    int ni = origToNode[oi];
	    int nj = origToNode[oj];

   	    edge_i.push_back(ni);
   	    edge_j.push_back(nj);
   	    
   	     edge_features.push_back({
   	     dca[idx],
   	     deltaR[idx],
   	     dca_sig[idx],
   	     cptopv[idx],
   	     pvtoPCA_i[idx],
   	     pvtoPCA_j[idx],
   	     dotprod_i[idx],
   	     dotprod_j[idx],
   	     pair_mom[idx],
   	     pair_invmass[idx]
   	     }); 
   	 }

   	std::vector<float> x_in_flat;
   	x_in_flat.reserve(track_features.size() * 14);
   	for (const auto& feat : track_features)
   	    x_in_flat.insert(x_in_flat.end(), feat.begin(), feat.end());

   	std::vector<float> edge_index_flat_f;
   	edge_index_flat_f.reserve(edge_i.size() * 2);
   	for (size_t k = 0; k < edge_i.size(); ++k) {
   	    edge_index_flat_f.push_back(static_cast<float>(edge_i[k]));
   	}

   	for (size_t k = 0; k < edge_j.size(); ++k) {
   	    edge_index_flat_f.push_back(static_cast<float>(edge_j[k]));
   	}

   	std::vector<float> edge_attr_flat;
   	edge_attr_flat.reserve(edge_features.size() * 10);
   	for (const auto& feat : edge_features)
   	    edge_attr_flat.insert(edge_attr_flat.end(), feat.begin(), feat.end());

   	 //std::cout << "track_features.size(): " << track_features.size() << "\n";
   	 //std::cout << "x_in_flat.size(): " << x_in_flat.size() << "\n";
   	 //std::cout << "edge_index_flat_f.size(): " << edge_index_flat_f.size() << "\n";
   	 //std::cout << "edge_attr_flat.size(): " << edge_attr_flat.size() << "\n";
   	 //std::cout << "edge_i.size(): " << edge_i.size() << "\n";
   	 //std::cout << "edge_features.size(): " << edge_features.size() << "\n";


   	   
   	// === 4. Set input names and feed data ===
   	std::vector<std::string> input_names_ = {"x_in", "edge_index", "edge_attr"};

   	std::vector<std::vector<int64_t>> input_shapes_ = {
   	 {1, static_cast<int64_t>(track_features.size()), 14},        // x_in
   	 {1, 2, static_cast<int64_t>(edge_i.size())},                // edge_index
   	 {1, static_cast<int64_t>(edge_features.size()), 10}         // edge_attr
   	};
   	   
   	std::vector<std::vector<float>> data_ = {
   	 x_in_flat,
   	 edge_index_flat_f,
   	 edge_attr_flat
   	};


   	std::vector<std::vector<float>> output = globalCache()->run(input_names_, data_, input_shapes_);

   	const std::vector<float>& sv_logits_flat = output[0];
   	const std::vector<float>& sv_sub_logits_flat = output[1];
   	const std::vector<float>& edge_logits_flat  = output[2]; // [E]

	//preds = edge_probs_flat;


// 	  std::cerr << "node_probs_flat size = " << node_probs_flat.size() << "\n";
//s	td::cerr << "edge_probs_flat size = " << edge_probs_flat.size() << "\n";
//
//f	loat minNP=1e9, maxNP=-1e9;
//f	or (float v : node_probs_flat) { minNP = std::min(minNP,v); maxNP = std::max(maxNP,v); }
//s	td::cerr << "node_probs range: [" << minNP << ", " << maxNP << "]\n";
//
//f	loat minEP=1e9, maxEP=-1e9;
//f	or (float v : edge_probs_flat) { minEP = std::min(minEP,v); maxEP = std::max(maxEP,v); }
//s	td::cerr << "edge_probs range: [" << minEP << ", " << maxEP << "]\n";
   	



//B	uild IVF - like
   	std::unordered_set<size_t> selected_indices;
   	for (size_t i = 0; i < alltracks.size(); ++i) {
   	    //const reco::Track& trk = alltracks[i];

   	    //double dz = trk.dz(pv.position());
   	    //if (
   	    //         (alltracks[i].pt() > 0.8) && 
   	    //         (std::abs(alltracks[i].eta())<2.5) //NEED TO REMAP TRACKS AND EDGES IF UNCOMMENTED
   	    //          && (std::abs(dz) < 0.3)
   	    // )
   	    //     {
   	    //     selected_indices.insert(i);

   	    //     }
   	    selected_indices.insert(i);
   	}

   	std::cout << "selected_indices..size(): " << selected_indices.size() << "\n";
   	std::cout << "alltracks.size(): " << alltracks.size() << "\n";
   	
   	//IVF is running on tracks passing threshold on GNN score
   	std::unordered_set<size_t> expanded_indices = selected_indices;  // will include nearby tracks too

	std::vector<size_t> expanded_vec(expanded_indices.begin(), expanded_indices.end());
	std::sort(expanded_vec.begin(), expanded_vec.end());
	
   	   //IVF is running on genMatch tracks
   	//std::unordered_set<size_t> expanded_indices(matched_indices.begin(), matched_indices.end());

   	//std::cout << "matched_indices..size(): " << matched_indices.size() << "\n";
   	//
   	std::vector<reco::TransientTrack> t_trks_SV;
   	std::vector<size_t> t_trks_SV_indices;

   	// Step 3: Fill t_trks_SV with the full expanded set
   	t_trks_SV.clear();  // if not already empty
   	t_trks_SV_indices.clear();

   	for (size_t idx : expanded_vec) {
   	    t_trks_SV.push_back(t_trks[idx]);
   	    t_trks_SV_indices.push_back(idx);
   	}

	  std::cout << "===== T_trks_SV  size=" << t_trks_SV.size() << " =====\n";
	  std::cout << "===== All tracks size=" << alltracks.size() << " =====\n";
	  std::cout << "===== valid tracks  size=" << trk_eta.size() << " =====\n";

	


	// ============================================================================
	// Vertexing from new model outputs:
	//   sv_logits_flat      : [N,3]   (node logits: [nonSV, PV, SV])
	//   sv_sub_logits_flat  : [N,3]   (subclass logits among SV-like: [B, C, CfromB])
	//   edge_logits_flat    : [E]     (edge logits; apply sigmoid)
	//
	// Assumes you already built t_trks_SV (size N) and the corresponding model inputs
	// (x_in_flat, edge_index_flat_f, edge_attr_flat, edge_i/edge_j etc.) for those N tracks.
	// Produces: comps (clusters of local node indices) and clustersTT (tracks per cluster)
	// that you can feed into your AVR stage.
	//
	// IMPORTANT: edge_index_flat_f layout is [2*E] = [src0..src(E-1), dst0..dst(E-1)]
	// ============================================================================
	
	// ------------------------------
	// 1) Sizes / constants
	// ------------------------------
	constexpr int C_NODE = 3; // node logits per track: [nonSV, PV, SV]
	constexpr int C_SUB  = 3; // subclass logits per node (B/C/CfromB)
	
	int N = (int)t_trks_SV.size();


       	// E from your edge construction. Prefer edge_i.size() if that's the number of edges you built.
	int E = (int)edge_i.size();
	
	// Sanity checks against model outputs
	assert((int)sv_logits_flat.size()     == N * C_NODE);
	assert((int)sv_sub_logits_flat.size() == N * C_SUB);
	assert((int)edge_logits_flat.size()   == E);
	assert((int)edge_index_flat_f.size()  == 2 * E);

	// ------------------------------
	// Tunables (scan these)
	// ------------------------------
	const float SeedSVcut     = 0.20f;   // seeds must be SV-like
	const float NodeSVmin     = 0.10f;   // allow low pSV nodes ONLY during expansion (not for seeding)
	const float PVVetoCut     = 0.50f;   // reject tracks with large PV probability from SV flow
	const float EdgeMin       = 0.30f;   // used only for bookkeeping / optional weak attach (keep as you had)
	const float EdgeStrongCC  = 0.20f;   // STRONG edges for connectivity (CC), scan 0.85..0.95
	const float EdgeStrongAtt = 0.20f;   // STRONG edges for attachment into an existing comp
	const int   MinClusterSize = 2;
	
	// Expansion rule: node joins comp if it has >=AttachMinStrong strong edges into that comp,
	// OR mean strong-edge score into comp >= AttachMinMean.
	const int   AttachMinStrong = 2;     // try 1 or 2
	const float AttachMinMean   = 0.30f; // try 0.85..0.92
	
	// Optional mutual topK pruning on strong edges (recommended; kills bridges)
	const bool  UseMutualTopK = true;
	const int   TopK = 8;               // keep topK strong neighbors per node (mutual only), try 3..8

	
	// ------------------------------
	// 2) Math helpers
	// ------------------------------
	auto sigmoid = [](float x) -> float {
	  // stable-ish sigmoid
	  if (x >= 0.f) {
	    float z = std::exp(-x);
	    return 1.f / (1.f + z);
	  } else {
	    float z = std::exp(x);
	    return z / (1.f + z);
	  }
	};
	
	auto softmaxNode3 = [&](int i, float out[3]) {
	  // sv_logits_flat layout: [N,3] = [nonSV, PV, SV]
	  float a = sv_logits_flat[i*3 + 0];
	  float b = sv_logits_flat[i*3 + 1];
	  float c = sv_logits_flat[i*3 + 2];
	  float m = std::max(a, std::max(b, c));
	  float ea = std::exp(a - m);
	  float eb = std::exp(b - m);
	  float ec = std::exp(c - m);
	  float s  = ea + eb + ec;
	  out[0] = ea/s; out[1] = eb/s; out[2] = ec/s;
	};
	
	auto softmaxSub3 = [&](int i, float out[3]) {
	  // sv_sub_logits_flat layout: [N,3]
	  float a = sv_sub_logits_flat[i*3 + 0];
	  float b = sv_sub_logits_flat[i*3 + 1];
	  float c = sv_sub_logits_flat[i*3 + 2];
	  float m = std::max(a, std::max(b, c));
	  float ea = std::exp(a - m);
	  float eb = std::exp(b - m);
	  float ec = std::exp(c - m);
	  float s  = ea + eb + ec;
	  out[0] = ea/s; out[1] = eb/s; out[2] = ec/s;
	};
	
	auto pSV = [&](int i) -> float {
	  float p[3]; softmaxNode3(i, p);
	  return p[2]; // class 2 is SV
	};

	auto pPV = [&](int i) -> float {
	  float p[3]; softmaxNode3(i, p);
	  return p[1]; // class 1 is PV
	};

	auto passSVGate = [&](int i, float svCut) -> bool {
	  return (pSV(i) > svCut) && (pPV(i) < PVVetoCut);
	};
	
	//auto pSubBest = [&](int i) -> float {
	//  float p[3]; softmaxSub3(i, p);
	//  return std::max({p[0], p[1], p[2]});
	//};
	
	auto subArgMax = [&](int i) -> int {
	  float p[3]; softmaxSub3(i, p);
	  int a = 0;
	  if (p[1] > p[a]) a = 1;
	  if (p[2] > p[a]) a = 2;
	  return a; // 0=B, 1=C, 2=CfromB (or however you define them)
	};
	
	// ------------------------------
	// 3) Edge index accessors
	// ------------------------------
	auto edgeSrc = [&](int e) -> int { return (int)edge_index_flat_f[e]; };
	auto edgeDst = [&](int e) -> int { return (int)edge_index_flat_f[E + e]; };
	auto pEdge   = [&](int e) -> float { return sigmoid(edge_logits_flat[e]); };

	// ------------------------------
	// Step 4-7 wrapped as a reusable builder, so we can scan structural cuts.
	// ------------------------------
	auto pack = [&](int a, int b) -> uint64_t {
	  return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
	};
	struct U64Hash { size_t operator()(uint64_t x) const noexcept { return std::hash<uint64_t>{}(x); } };

	auto buildComponents = [&](float seedCut, float nodeMinCut, float edgeMinCut) {
	  std::vector<char> isSeed(N, 0);
	  std::vector<char> isEligible(N, 0);
	  for (int i = 0; i < N; ++i) {
	    if (passSVGate(i, seedCut)) isSeed[i] = 1;
	    if (passSVGate(i, nodeMinCut)) isEligible[i] = 1;
	  }

	  std::vector<std::vector<int>> adjStrongRaw(N);
	  std::unordered_map<uint64_t, float, U64Hash> edgeScore;
	  edgeScore.reserve((size_t)E * 2);

	  for (int e = 0; e < E; ++e) {
	    int u = edgeSrc(e);
	    int v = edgeDst(e);
	    if ((unsigned)u >= (unsigned)N || (unsigned)v >= (unsigned)N) continue;
	    if (!isEligible[u] || !isEligible[v]) continue;

	    float pe = pEdge(e);
	    if (!std::isfinite(pe)) continue;
	    if (pe < edgeMinCut) continue;

	    edgeScore[pack(u,v)] = pe;
	    edgeScore[pack(v,u)] = pe;

	    if (pe >= EdgeStrongCC) {
	      adjStrongRaw[u].push_back(v);
	      adjStrongRaw[v].push_back(u);
	    }
	  }

	  std::vector<std::vector<int>> adjStrong(N);
	  if (UseMutualTopK) {
	    std::vector<std::vector<int>> topNbrs(N);
	    topNbrs.reserve(N);
	    for (int u = 0; u < N; ++u) {
	      auto &nb = adjStrongRaw[u];
	      if (nb.empty()) continue;
	      std::sort(nb.begin(), nb.end(), [&](int a, int b){
	        float sa = 0.f, sb = 0.f;
	        auto ita = edgeScore.find(pack(u,a)); if (ita != edgeScore.end()) sa = ita->second;
	        auto itb = edgeScore.find(pack(u,b)); if (itb != edgeScore.end()) sb = itb->second;
	        return sa > sb;
	      });
	      if ((int)nb.size() > TopK) nb.resize(TopK);
	      topNbrs[u] = nb;
	    }

	    std::unordered_set<uint64_t, U64Hash> keepEdge;
	    keepEdge.reserve((size_t)N * (size_t)TopK * 2);
	    for (int u = 0; u < N; ++u) for (int v : topNbrs[u]) keepEdge.insert(pack(u,v));
	    for (int u = 0; u < N; ++u) {
	      for (int v : topNbrs[u]) {
	        if (keepEdge.find(pack(v,u)) == keepEdge.end()) continue;
	        adjStrong[u].push_back(v);
	      }
	    }
	  } else {
	    adjStrong.swap(adjStrongRaw);
	  }

	  for (int u = 0; u < N; ++u) {
	    auto &nb = adjStrong[u];
	    if (nb.empty()) continue;
	    std::sort(nb.begin(), nb.end());
	    nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
	  }

	  std::vector<char> visited(N, 0);
	  std::vector<std::vector<int>> localComps;
	  localComps.reserve(N);
	  std::vector<int> stack;
	  stack.reserve(512);

	  for (int start = 0; start < N; ++start) {
	    if (!isSeed[start] || visited[start] || adjStrong[start].empty()) continue;
	    visited[start] = 1;
	    localComps.emplace_back();
	    auto &comp = localComps.back();
	    comp.reserve(64);
	    stack.clear();
	    stack.push_back(start);
	    while (!stack.empty()) {
	      int u = stack.back();
	      stack.pop_back();
	      comp.push_back(u);
	      for (int v : adjStrong[u]) {
	        if (!isEligible[v] || visited[v]) continue;
	        visited[v] = 1;
	        stack.push_back(v);
	      }
	    }
	  }

	  std::vector<int> compId(N, -1);
	  for (int ci = 0; ci < (int)localComps.size(); ++ci) for (int u : localComps[ci]) compId[u] = ci;

	  std::vector<std::vector<int>> adjAttach(N);
	  for (int e = 0; e < E; ++e) {
	    int u = edgeSrc(e);
	    int v = edgeDst(e);
	    if ((unsigned)u >= (unsigned)N || (unsigned)v >= (unsigned)N) continue;
	    if (!isEligible[u] || !isEligible[v]) continue;
	    float pe = pEdge(e);
	    if (!std::isfinite(pe) || pe < EdgeStrongAtt) continue;
	    adjAttach[u].push_back(v);
	    adjAttach[v].push_back(u);
	  }
	  for (int u = 0; u < N; ++u) {
	    auto &nb = adjAttach[u];
	    if (nb.empty()) continue;
	    std::sort(nb.begin(), nb.end());
	    nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
	  }

	  for (int ci = 0; ci < (int)localComps.size(); ++ci) {
	    auto &comp = localComps[ci];
	    if ((int)comp.size() < MinClusterSize) continue;
	    static std::vector<int> mark;
	    static int stamp = 1;
	    if ((int)mark.size() != N) mark.assign(N, 0);
	    ++stamp;
	    for (int u : comp) mark[u] = stamp;

	    bool changed = true;
	    int iter = 0;
	    const int MaxExpandIters = 6;
	    while (changed && iter++ < MaxExpandIters) {
	      changed = false;
	      std::vector<int> candidates;
	      candidates.reserve(comp.size() * 4);
	      for (int u : comp) {
	        for (int v : adjAttach[u]) {
	          if (!isEligible[v] || mark[v] == stamp) continue;
	          candidates.push_back(v);
	        }
	      }
	      if (candidates.empty()) break;
	      std::sort(candidates.begin(), candidates.end());
	      candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

	      for (int v : candidates) {
	        int strongCnt = 0;
	        float sum = 0.f;
	        int cnt = 0;
	        for (int nb : adjAttach[v]) {
	          if (mark[nb] != stamp) continue;
	          auto it = edgeScore.find(pack(v, nb));
	          if (it == edgeScore.end()) continue;
	          float pe = it->second;
	          if (pe < EdgeStrongAtt) continue;
	          strongCnt++;
	          sum += pe;
	          cnt++;
	        }
	        float mean = (cnt > 0) ? (sum / float(cnt)) : 0.f;
	        if (strongCnt >= AttachMinStrong || mean >= AttachMinMean) {
	          int old = compId[v];
	          if (old >= 0) {
	            auto &oldComp = localComps[old];
	            oldComp.erase(std::remove(oldComp.begin(), oldComp.end(), v), oldComp.end());
	          }
	          comp.push_back(v);
	          mark[v] = stamp;
	          compId[v] = ci;
	          changed = true;
	        }
	      }
	    }
	  }

	  //auto hasSVcore = [&](const std::vector<int> &comp) -> bool {
	  //  float maxPSV = 0.f;
	  //  int cnt30 = 0;
	  //  for (int u : comp) {
	  //    float ps = pSV(u);
	  //    if (ps > maxPSV) maxPSV = ps;
	  //    if (ps > 0.30f) cnt30++;
	  //  }
	  //  float frac30 = comp.empty() ? 0.f : float(cnt30) / float(comp.size());
	  //  return (maxPSV > 0.70f) && (frac30 > 0.40f);
	  //};

	  localComps.erase(
	    std::remove_if(localComps.begin(), localComps.end(), [&](auto const& c){
	      if ((int)c.size() < MinClusterSize) return true;
	      //if (!hasSVcore(c)) return true;
	      return false;
	    }),
	    localComps.end()
	  );
	  return localComps;
	};

	std::vector<std::vector<int>> comps = buildComponents(SeedSVcut, NodeSVmin, EdgeMin);
	
	
	
	
	//// ------------------------------
	//// 8) OPTIONAL: split each comp by sublabel if you want separate B/C/CfromB vertices
	//// Do this only if you see mixed comps; otherwise skip.
	//// ------------------------------
	//const bool SplitBySubLabel = false;
	//
	//if (SplitBySubLabel) {
	//  std::vector<std::vector<int>> comps2;
	//  comps2.reserve(comps.size());
	//
	//  for (auto const& comp : comps) {
	//    std::vector<int> g0, g1, g2;
	//    g0.reserve(comp.size()); g1.reserve(comp.size()); g2.reserve(comp.size());
	//
	//    for (int u : comp) {
	//      int a = subArgMax(u);
	//      if      (a == 0) g0.push_back(u);
	//      else if (a == 1) g1.push_back(u);
	//      else             g2.push_back(u);
	//    }
	//
	//    if ((int)g0.size() >= MinClusterSize) comps2.push_back(std::move(g0));
	//    if ((int)g1.size() >= MinClusterSize) comps2.push_back(std::move(g1));
	//    if ((int)g2.size() >= MinClusterSize) comps2.push_back(std::move(g2));
	//  }
	//
	//  comps.swap(comps2);
	//}

	//IF mapping needed
	auto toTrkIdx = [&](int idx) -> int {
	  return (int)t_trks_SV_indices[idx]; // must exist and be size N
	};




//Writing model outputs to histograms after genmatching for sig/bkg
//
sv_score_sig.reserve(N);
sv_score_bkg.reserve(N);
edge_score_sig.reserve(E);
edge_score_bkg.reserve(E);

for (int node = 0; node < N; ++node) {
    const int trk = toTrkIdx(node);
    const bool is_sig = (truthByTrack.find(trk) != truthByTrack.end());

    double score = static_cast<double>(pSV(node));
    score = std::clamp(score, 0.0, 1.0);

    (is_sig ? sv_score_sig : sv_score_bkg).push_back(score);
 }  

for (int e = 0; e < E; ++e) {
    const int u = edge_i[e];
    const int v = edge_j[e];

    const int tu = toTrkIdx(u);
    const int tv = toTrkIdx(v);

    bool is_sig_edge = false;
    const auto itu = truthByTrack.find(tu);
    const auto itv = truthByTrack.find(tv);
    if (itu != truthByTrack.end() && itv != truthByTrack.end()) {
      is_sig_edge = (itu->second.first == itv->second.first) &&
                    (itu->second.second == itv->second.second);
    }

    double score = static_cast<double>(sigmoid(edge_logits_flat[e]));
    score = std::clamp(score, 0.0, 1.0);

    (is_sig_edge ? edge_score_sig : edge_score_bkg).push_back(score);
  }

// ============================================================================
// Matching / printing with new model outputs:
//   - truth label exists in [0..6]; you care about truth labels {2,3,4}
//   - sub_logits argmax is in {0,1,2} and should correspond to truth {2,3,4}
//     mapping: truth 2->sub 0, truth 3->sub 1, truth 4->sub 2
//
// "SV match" definition (per truth label 2/3/4 separately):
//   A hadidx is matched for label L if there exists SOME component where
//   >=2 tracks from that hadidx have truth label L AND predicted sublabel == map(L).
//
// "Fake" definition (per predicted sublabel s=0/1/2):
//   In a component, if there are >=2 tracks with NO truth match (not in truthByTrack)
//   and they share predicted sublabel s, count one fake vertex instance for s.
// ============================================================================

// ------------------------------
// Helpers: predicted pSV and predicted sublabel
// ------------------------------
// truth label -> expected sublabel
auto truthToSub = [&](int truthLab) -> int {
  // You stated: truth 2,3,4 correspond to sub 0,1,2
  if (truthLab == 2) return 0;
  if (truthLab == 3) return 1;
  if (truthLab == 4) return 2;
  return -1;
};

// For printing: predicted node class as argmax over [nonSV, PV, SV]
auto modelNodeClass = [&](int idx) -> int {
  float p[3]; softmaxNode3(idx, p);
  int a = 0;
  if (p[1] > p[a]) a = 1;
  if (p[2] > p[a]) a = 2;
  return a; // 0=nonSV, 1=PV, 2=SV
};

// ------------------------------
// (A) Print comp details: trkidx modelSV modelSubSV truthlabel truthhadidx
// ------------------------------
std::unordered_set<int> matchedTruthTracks;
matchedTruthTracks.reserve(1024);

auto printCompDetails = [&](const std::vector<int>& comp, int compnum) {
  std::cout << "===== Component " << compnum << " size=" << comp.size() << " =====\n";
  std::cout << "trkIdx  nodeCls  pSV  pPV  modelSubSV  truthLabel  truthHadidx\n";
  std::cout << "----------------------------------------------------------------\n";

  for (int idx : comp) {
    const int trk = toTrkIdx(idx);
    const int mNode = modelNodeClass(idx);
    const float psv = pSV(idx);
    const float ppv = pPV(idx);
    const int mSu = subArgMax(idx)+2;

    auto jt = truthByTrack.find(trk);
    if (jt != truthByTrack.end()) {
      matchedTruthTracks.insert(trk);
      const int tLab = jt->second.first;
      const int tHid = jt->second.second;

      std::cout << std::setw(5) << trk << "  "
                << std::setw(7) << mNode << "  "
                << std::setw(3) << std::fixed << std::setprecision(2) << psv << "  "
                << std::setw(3) << std::fixed << std::setprecision(2) << ppv << "  "
                << std::setw(10) << mSu << "  "
                << std::setw(10) << tLab << "  "
                << std::setw(11) << tHid << "\n";
    } else {
      std::cout << std::setw(5) << trk << "  "
                << std::setw(7) << mNode << "  "
                << std::setw(3) << std::fixed << std::setprecision(2) << psv << "  "
                << std::setw(3) << std::fixed << std::setprecision(2) << ppv << "  "
                << std::setw(10) << mSu << "  "
                << std::setw(10) << "NA" << "  "
                << std::setw(11) << "NA" << "\n";
    }
  }

  std::cout << "\n";
};

// Print comps
int compnum = 0;
for (auto const& comp : comps) {
  printCompDetails(comp, compnum++);
}

// Unmatched truth tracks (same as your old helper; updated to use trk idx set)
auto printUnmatchedTruth = [&]() {
  std::cout << "\n";
  std::cout << "==================== UNMATCHED GENMATCH TRACKS ====================\n";
  std::cout << "trkIdx  truthLabel  truthHadidx\n";
  std::cout << "--------------------------------------------------------------------\n";

  if (!genmatch) {
    std::cout << "(no genmatch entry for this event)\n\n";
    return;
  }

  int nUnmatched = 0;
  for (size_t k = 0; k < genmatch->indices.size(); ++k) {
    const int trk = genmatch->indices[k];
    if (matchedTruthTracks.find(trk) != matchedTruthTracks.end()) continue;

    ++nUnmatched;
    std::cout << std::setw(5) << trk << "  "
              << std::setw(10) << genmatch->labels[k] << "  "
              << std::setw(11) << genmatch->hadidx[k] << "\n";
  }

  if (nUnmatched == 0) std::cout << "(none)\n";
  std::cout << "====================================================================\n\n";
};

printUnmatchedTruth();
// ------------------------------
// (B) Denominator: unique hadidx per truth label (2/3/4) in the truth
// ------------------------------
constexpr int C_TRUTH = 7; // truth labels 0..6
std::array<std::unordered_set<int>, C_TRUTH> totalHadidxByTruthLabel;
for (auto& s : totalHadidxByTruthLabel) s.reserve(256);

if (genmatch) {
  for (size_t k = 0; k < genmatch->indices.size(); ++k) {
    const int lab = genmatch->labels[k];
    const int hid = genmatch->hadidx[k];
    if (hid < 0) continue;
    if (lab == 2 || lab == 3 || lab == 4) {
      totalHadidxByTruthLabel[lab].insert(hid);
    }
  }
}

// ------------------------------
// (C) Numerator: unique hadidx per truth label matched by your comps
// Match definition: in SOME component, >=2 tracks from same hadidx AND truth label L,
// AND predicted sublabel == truthToSub(L).
// Optionally require modelSV=1 or pSV > cut.
// ------------------------------
const bool RequireModelSV = true;
const float pSVcut = 0.10f;

std::array<std::unordered_set<int>, C_TRUTH> matchedHadidxByTruthLabel;
for (auto& s : matchedHadidxByTruthLabel) s.reserve(256);

for (auto const& comp : comps) {
  // per comp counts by hadidx for each label 2/3/4
  std::unordered_map<int,int> cnt2, cnt3, cnt4;
  cnt2.reserve(64); cnt3.reserve(64); cnt4.reserve(64);

  for (int idx : comp) {
    const int trk = toTrkIdx(idx);

    auto it = truthByTrack.find(trk);
    if (it == truthByTrack.end()) continue;

    const int lab = it->second.first;
    const int hid = it->second.second;
    if (hid < 0) continue;

    if (!(lab == 2 || lab == 3 || lab == 4)) continue;

    // predicted requirements
    if (RequireModelSV) {
      if (!passSVGate(idx, pSVcut)) continue;
    }

    const int predSub = subArgMax(idx);
    const int wantSub = truthToSub(lab);
    if (wantSub < 0) continue;

    // require predicted sublabel corresponds to this truth label
    if (predSub != wantSub) continue;

    if (lab == 2) ++cnt2[hid];
    else if (lab == 3) ++cnt3[hid];
    else               ++cnt4[hid];
  }

  for (auto const& [hid, n] : cnt2) if (n >= 2) matchedHadidxByTruthLabel[2].insert(hid);
  for (auto const& [hid, n] : cnt3) if (n >= 2) matchedHadidxByTruthLabel[3].insert(hid);
  for (auto const& [hid, n] : cnt4) if (n >= 2) matchedHadidxByTruthLabel[4].insert(hid);
}

// ------------------------------
// (D) Print efficiency per truth label
// ------------------------------
auto printEff = [&](int lab) {
  const size_t denom = totalHadidxByTruthLabel[lab].size();
  const size_t num   = matchedHadidxByTruthLabel[lab].size();
  const float eff    = (denom == 0 ? 0.f : float(num) / float(denom));

  if(lab==2)
  {
	nHad_B.push_back((int)denom);
	eff_B.push_back(eff);
  }

  if(lab==3)
  {
        nHad_BtoC.push_back((int)denom);
        eff_BtoC.push_back(eff);
  }
 
  if(lab==4)
  {
        nHad_C.push_back((int)denom);
        eff_C.push_back(eff);
  }

  if (denom > 0) {
    std::cout << "truth label " << lab
              << " hadidxMatched=" << num
              << " hadidxTotal=" << denom
              << " efficiency=" << std::fixed << std::setprecision(3) << eff
              << "\n";
  }
};

std::cout << "=== Vertex (hadidx) match efficiency using predicted sublabels ===\n";
printEff(2);
printEff(3);
printEff(4);

// ------------------------------
// (E) Fake vertices by predicted sublabel (same idea as before)
// Definition: in a comp, if >=2 tracks with NO truth entry share the same predicted sublabel,
// count a fake vertex instance for that sublabel. Optionally require SV-like.
// ------------------------------
std::array<int, 3> fakeVerticesByPredSub{}; // init 0
std::array<int, 3> fakeTracksByPredSub{};   // init 0

for (auto const& comp : comps) {
  std::array<int, 3> unmatchedCountBySub{}; // per comp

  for (int idx : comp) {
    const int trk = toTrkIdx(idx);

    const bool hasTruth = (truthByTrack.find(trk) != truthByTrack.end());
    if (hasTruth) continue;

    if (RequireModelSV) {
      if (!passSVGate(idx, pSVcut)) continue;
    }

    const int sub = subArgMax(idx);
    if (sub >= 0 && sub < 3) unmatchedCountBySub[sub]++;
  }

  for (int s = 0; s < 3; ++s) {
    if (unmatchedCountBySub[s] >= 2) {
      fakeVerticesByPredSub[s] += 1;
      fakeTracksByPredSub[s]   += unmatchedCountBySub[s];
    }
  }
}

std::cout << "=== Fake vertices (unmatched tracks): >=2 unmatched tracks with same predicted sublabel ===\n";
std::cout << "predSub 0 (truth2): fakeVertexInstances=" << fakeVerticesByPredSub[0]
          << " fakeTracksUsed=" << fakeTracksByPredSub[0] << "\n";
std::cout << "predSub 1 (truth3): fakeVertexInstances=" << fakeVerticesByPredSub[1]
          << " fakeTracksUsed=" << fakeTracksByPredSub[1] << "\n";
std::cout << "predSub 2 (truth4): fakeVertexInstances=" << fakeVerticesByPredSub[2]
          << " fakeTracksUsed=" << fakeTracksByPredSub[2] << "\n";

fake_B.push_back(fakeVerticesByPredSub[0]);
fake_BtoC.push_back(fakeVerticesByPredSub[1]);
fake_C.push_back(fakeVerticesByPredSub[2]);

// ---------------------------------------------------------------------------
// Operating-point scan in one pass:
//   - Scans pSV and PV-veto thresholds used in matching/fake bookkeeping
//   - Stores aggregate numerators/denominators and fake counts in 2D histograms
//   - Final efficiency map can be derived offline as num/den per bin
// ---------------------------------------------------------------------------
static bool scanHistsInit = false;
static TH1F* h_scan_seed_num_B = nullptr;
static TH1F* h_scan_seed_den_B = nullptr;
static TH1F* h_scan_seed_num_BtoC = nullptr;
static TH1F* h_scan_seed_den_BtoC = nullptr;
static TH1F* h_scan_seed_num_C = nullptr;
static TH1F* h_scan_seed_den_C = nullptr;
static TH1F* h_scan_seed_fake_B = nullptr;
static TH1F* h_scan_seed_fake_BtoC = nullptr;
static TH1F* h_scan_seed_fake_C = nullptr;
static TH1F* h_scan_seed_events = nullptr;
static TH1F* h_scan_node_num_B = nullptr;
static TH1F* h_scan_node_den_B = nullptr;
static TH1F* h_scan_node_num_BtoC = nullptr;
static TH1F* h_scan_node_den_BtoC = nullptr;
static TH1F* h_scan_node_num_C = nullptr;
static TH1F* h_scan_node_den_C = nullptr;
static TH1F* h_scan_node_fake_B = nullptr;
static TH1F* h_scan_node_fake_BtoC = nullptr;
static TH1F* h_scan_node_fake_C = nullptr;
static TH1F* h_scan_node_events = nullptr;
static TH1F* h_scan_edge_num_B = nullptr;
static TH1F* h_scan_edge_den_B = nullptr;
static TH1F* h_scan_edge_num_BtoC = nullptr;
static TH1F* h_scan_edge_den_BtoC = nullptr;
static TH1F* h_scan_edge_num_C = nullptr;
static TH1F* h_scan_edge_den_C = nullptr;
static TH1F* h_scan_edge_fake_B = nullptr;
static TH1F* h_scan_edge_fake_BtoC = nullptr;
static TH1F* h_scan_edge_fake_C = nullptr;
static TH1F* h_scan_edge_events = nullptr;
static TH1F* h_b_diag_stage_sum = nullptr;
static TH1F* h_btoc_diag_stage_sum = nullptr;
static TH1F* h_c_diag_stage_sum = nullptr;
static TH1F* h_c_diag_event_counter = nullptr;
static TH1F* h_b_diag_event_counter_with_b = nullptr;
static TH1F* h_btoc_diag_event_counter_with_btoc = nullptr;
static TH1F* h_c_diag_event_counter_with_c = nullptr;

if (!scanHistsInit) {
  edm::Service<TFileService> fs;
  h_scan_seed_num_B    = fs->make<TH1F>("scan_seed_num_B", "Matched hadidx (truth=2);SeedSVcut;count", 19, 0.025, 0.975);
  h_scan_seed_den_B    = fs->make<TH1F>("scan_seed_den_B", "Total hadidx (truth=2);SeedSVcut;count", 19, 0.025, 0.975);
  h_scan_seed_num_BtoC = fs->make<TH1F>("scan_seed_num_BtoC", "Matched hadidx (truth=3);SeedSVcut;count", 19, 0.025, 0.975);
  h_scan_seed_den_BtoC = fs->make<TH1F>("scan_seed_den_BtoC", "Total hadidx (truth=3);SeedSVcut;count", 19, 0.025, 0.975);
  h_scan_seed_num_C    = fs->make<TH1F>("scan_seed_num_C", "Matched hadidx (truth=4);SeedSVcut;count", 19, 0.025, 0.975);
  h_scan_seed_den_C    = fs->make<TH1F>("scan_seed_den_C", "Total hadidx (truth=4);SeedSVcut;count", 19, 0.025, 0.975);
  h_scan_seed_fake_B   = fs->make<TH1F>("scan_seed_fake_B", "Fake vertices (predSub=0);SeedSVcut;count", 19, 0.025, 0.975);
  h_scan_seed_fake_BtoC= fs->make<TH1F>("scan_seed_fake_BtoC", "Fake vertices (predSub=1);SeedSVcut;count", 19, 0.025, 0.975);
  h_scan_seed_fake_C   = fs->make<TH1F>("scan_seed_fake_C", "Fake vertices (predSub=2);SeedSVcut;count", 19, 0.025, 0.975);
  h_scan_seed_events   = fs->make<TH1F>("scan_seed_events", "Event counter;SeedSVcut;count", 19, 0.025, 0.975);

  h_scan_node_num_B    = fs->make<TH1F>("scan_node_num_B", "Matched hadidx (truth=2);NodeSVmin;count", 19, 0.025, 0.975);
  h_scan_node_den_B    = fs->make<TH1F>("scan_node_den_B", "Total hadidx (truth=2);NodeSVmin;count", 19, 0.025, 0.975);
  h_scan_node_num_BtoC = fs->make<TH1F>("scan_node_num_BtoC", "Matched hadidx (truth=3);NodeSVmin;count", 19, 0.025, 0.975);
  h_scan_node_den_BtoC = fs->make<TH1F>("scan_node_den_BtoC", "Total hadidx (truth=3);NodeSVmin;count", 19, 0.025, 0.975);
  h_scan_node_num_C    = fs->make<TH1F>("scan_node_num_C", "Matched hadidx (truth=4);NodeSVmin;count", 19, 0.025, 0.975);
  h_scan_node_den_C    = fs->make<TH1F>("scan_node_den_C", "Total hadidx (truth=4);NodeSVmin;count", 19, 0.025, 0.975);
  h_scan_node_fake_B   = fs->make<TH1F>("scan_node_fake_B", "Fake vertices (predSub=0);NodeSVmin;count", 19, 0.025, 0.975);
  h_scan_node_fake_BtoC= fs->make<TH1F>("scan_node_fake_BtoC", "Fake vertices (predSub=1);NodeSVmin;count", 19, 0.025, 0.975);
  h_scan_node_fake_C   = fs->make<TH1F>("scan_node_fake_C", "Fake vertices (predSub=2);NodeSVmin;count", 19, 0.025, 0.975);
  h_scan_node_events   = fs->make<TH1F>("scan_node_events", "Event counter;NodeSVmin;count", 19, 0.025, 0.975);

  h_scan_edge_num_B    = fs->make<TH1F>("scan_edge_num_B", "Matched hadidx (truth=2);EdgeMin;count", 19, 0.025, 0.975);
  h_scan_edge_den_B    = fs->make<TH1F>("scan_edge_den_B", "Total hadidx (truth=2);EdgeMin;count", 19, 0.025, 0.975);
  h_scan_edge_num_BtoC = fs->make<TH1F>("scan_edge_num_BtoC", "Matched hadidx (truth=3);EdgeMin;count", 19, 0.025, 0.975);
  h_scan_edge_den_BtoC = fs->make<TH1F>("scan_edge_den_BtoC", "Total hadidx (truth=3);EdgeMin;count", 19, 0.025, 0.975);
  h_scan_edge_num_C    = fs->make<TH1F>("scan_edge_num_C", "Matched hadidx (truth=4);EdgeMin;count", 19, 0.025, 0.975);
  h_scan_edge_den_C    = fs->make<TH1F>("scan_edge_den_C", "Total hadidx (truth=4);EdgeMin;count", 19, 0.025, 0.975);
  h_scan_edge_fake_B   = fs->make<TH1F>("scan_edge_fake_B", "Fake vertices (predSub=0);EdgeMin;count", 19, 0.025, 0.975);
  h_scan_edge_fake_BtoC= fs->make<TH1F>("scan_edge_fake_BtoC", "Fake vertices (predSub=1);EdgeMin;count", 19, 0.025, 0.975);
  h_scan_edge_fake_C   = fs->make<TH1F>("scan_edge_fake_C", "Fake vertices (predSub=2);EdgeMin;count", 19, 0.025, 0.975);
  h_scan_edge_events   = fs->make<TH1F>("scan_edge_events", "Event counter;EdgeMin;count", 19, 0.025, 0.975);

  auto setDiagBinLabels = [&](TH1F* h) {
    h->GetXaxis()->SetBinLabel(1, "total_C");
    h->GetXaxis()->SetBinLabel(2, "Had_with2tracks");
    h->GetXaxis()->SetBinLabel(3, "Had_with2tracks_cuts");
    h->GetXaxis()->SetBinLabel(4, "Had_with2tracks_cuts_comp");
    h->GetXaxis()->SetBinLabel(5, "Had_with2tracks_cuts_sublabel");
    h->GetXaxis()->SetBinLabel(6, "Had_finalselection");
  };

  h_b_diag_stage_sum = fs->make<TH1F>("b_diag_stage_sum", "B hadron stage sums;stage;count", 6, 0.5, 6.5);
  setDiagBinLabels(h_b_diag_stage_sum);
  h_btoc_diag_stage_sum = fs->make<TH1F>("btoc_diag_stage_sum", "BtoC hadron stage sums;stage;count", 6, 0.5, 6.5);
  setDiagBinLabels(h_btoc_diag_stage_sum);
  h_c_diag_stage_sum = fs->make<TH1F>("c_diag_stage_sum", "C hadron stage sums;stage;count", 6, 0.5, 6.5);
  setDiagBinLabels(h_c_diag_stage_sum);

  h_c_diag_event_counter = fs->make<TH1F>("c_diag_event_counter", "C diagnostics event counter;bin;count", 1, 0.5, 1.5);
  h_b_diag_event_counter_with_b = fs->make<TH1F>("b_diag_event_counter_with_b", "B diagnostics events with B hadidx;bin;count", 1, 0.5, 1.5);
  h_btoc_diag_event_counter_with_btoc = fs->make<TH1F>("btoc_diag_event_counter_with_btoc", "BtoC diagnostics events with BtoC hadidx;bin;count", 1, 0.5, 1.5);
  h_c_diag_event_counter_with_c = fs->make<TH1F>("c_diag_event_counter_with_c", "C diagnostics events with C hadidx;bin;count", 1, 0.5, 1.5);
  scanHistsInit = true;
}

h_c_diag_event_counter->Fill(1.0);

auto fillFlavorDiag = [&](int truthLabel, TH1F* stageHist, TH1F* hasFlavorEvtHist) {
  std::unordered_map<int, int> truthTrackCount;
  std::unordered_map<int, int> gateTrackCount;
  std::unordered_map<int, int> compGateTrackCount;
  std::unordered_map<int, int> gateSubTrackCount;
  truthTrackCount.reserve(256);
  gateTrackCount.reserve(256);
  compGateTrackCount.reserve(256);
  gateSubTrackCount.reserve(256);

  for (auto const& [trk, lh] : truthByTrack) {
    const int lab = lh.first;
    const int hid = lh.second;
    if (lab == truthLabel && hid >= 0) truthTrackCount[hid] += 1;
  }

  for (int idx = 0; idx < N; ++idx) {
    const int trk = toTrkIdx(idx);
    auto it = truthByTrack.find(trk);
    if (it == truthByTrack.end()) continue;
    const int lab = it->second.first;
    const int hid = it->second.second;
    if (lab != truthLabel || hid < 0) continue;
    if (passSVGate(idx, pSVcut)) {
      gateTrackCount[hid] += 1;
      if (subArgMax(idx) == truthToSub(truthLabel)) gateSubTrackCount[hid] += 1;
    }
  }

  for (auto const& comp : comps) {
    for (int idx : comp) {
      const int trk = toTrkIdx(idx);
      auto it = truthByTrack.find(trk);
      if (it == truthByTrack.end()) continue;
      const int lab = it->second.first;
      const int hid = it->second.second;
      if (lab != truthLabel || hid < 0) continue;
      if (passSVGate(idx, pSVcut)) compGateTrackCount[hid] += 1;
    }
  }

  int truthGe2 = 0;
  for (auto const& [hid, cnt] : truthTrackCount) if (cnt >= 2) truthGe2++;
  int gateGe2 = 0;
  for (auto const& [hid, cnt] : gateTrackCount) if (cnt >= 2) gateGe2++;
  int compGateGe2 = 0;
  for (auto const& [hid, cnt] : compGateTrackCount) if (cnt >= 2) compGateGe2++;
  int gateSubGe2 = 0;
  for (auto const& [hid, cnt] : gateSubTrackCount) if (cnt >= 2) gateSubGe2++;

  const int totalHadidx = static_cast<int>(totalHadidxByTruthLabel[truthLabel].size());
  const int finalMatched = static_cast<int>(matchedHadidxByTruthLabel[truthLabel].size());

  stageHist->Fill(1.0, static_cast<double>(totalHadidx));
  stageHist->Fill(2.0, static_cast<double>(truthGe2));
  stageHist->Fill(3.0, static_cast<double>(gateGe2));
  stageHist->Fill(4.0, static_cast<double>(compGateGe2));
  stageHist->Fill(5.0, static_cast<double>(gateSubGe2));
  stageHist->Fill(6.0, static_cast<double>(finalMatched));
  if (totalHadidx > 0) hasFlavorEvtHist->Fill(1.0);
};

fillFlavorDiag(2, h_b_diag_stage_sum, h_b_diag_event_counter_with_b);
fillFlavorDiag(3, h_btoc_diag_stage_sum, h_btoc_diag_event_counter_with_btoc);
fillFlavorDiag(4, h_c_diag_stage_sum, h_c_diag_event_counter_with_c);

auto passSVGateCut = [&](int i, float svCut, float pvCut) -> bool {
  return (pSV(i) > svCut) && (pPV(i) < pvCut);
};

const std::vector<float> scanSVcuts = {
  0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.35f, 0.40f, 0.45f, 0.50f,
  0.55f, 0.60f, 0.65f, 0.70f, 0.75f, 0.80f, 0.85f, 0.90f, 0.95f
};

auto evaluateComps = [&](const std::vector<std::vector<int>>& compsIn, float svCut, float pvCut) {
  std::array<float,3> num = {{0.f, 0.f, 0.f}};   // truth labels 2/3/4 mapped to [0/1/2]
  std::array<float,3> den = {{
    static_cast<float>(totalHadidxByTruthLabel[2].size()),
    static_cast<float>(totalHadidxByTruthLabel[3].size()),
    static_cast<float>(totalHadidxByTruthLabel[4].size())
  }};
  std::array<float,3> fake = {{0.f, 0.f, 0.f}};

  std::array<std::unordered_set<int>, C_TRUTH> matchedScan;
  for (auto& s : matchedScan) s.reserve(256);

  for (auto const& comp : compsIn) {
    std::unordered_map<int,int> cnt2, cnt3, cnt4;
    cnt2.reserve(64); cnt3.reserve(64); cnt4.reserve(64);
    std::array<int, 3> unmatchedCountBySub{};

    for (int idx : comp) {
      if (!passSVGateCut(idx, svCut, pvCut)) continue;
      const int trk = toTrkIdx(idx);
      auto itTruth = truthByTrack.find(trk);
      if (itTruth != truthByTrack.end()) {
        const int lab = itTruth->second.first;
        const int hid = itTruth->second.second;
        if (hid < 0) continue;
        if (!(lab == 2 || lab == 3 || lab == 4)) continue;
        const int predSub = subArgMax(idx);
        const int wantSub = truthToSub(lab);
        if (wantSub < 0 || predSub != wantSub) continue;
        if (lab == 2) ++cnt2[hid];
        else if (lab == 3) ++cnt3[hid];
        else ++cnt4[hid];
      } else {
        const int sub = subArgMax(idx);
        if (sub >= 0 && sub < 3) unmatchedCountBySub[sub]++;
      }
    }

    for (auto const& [hid, n] : cnt2) if (n >= 2) matchedScan[2].insert(hid);
    for (auto const& [hid, n] : cnt3) if (n >= 2) matchedScan[3].insert(hid);
    for (auto const& [hid, n] : cnt4) if (n >= 2) matchedScan[4].insert(hid);
    for (int s = 0; s < 3; ++s) if (unmatchedCountBySub[s] >= 2) fake[s] += 1.f;
  }

  num[0] = static_cast<float>(matchedScan[2].size());
  num[1] = static_cast<float>(matchedScan[3].size());
  num[2] = static_cast<float>(matchedScan[4].size());
  return std::make_tuple(num, den, fake);
};

// Structural scans: rebuild components while scanning SeedSVcut / NodeSVmin / EdgeMin.
for (float seedCutScan : scanSVcuts) {
  auto compsSeed = buildComponents(seedCutScan, NodeSVmin, EdgeMin);
  auto [num, den, fake] = evaluateComps(compsSeed, pSVcut, PVVetoCut);
  h_scan_seed_num_B->Fill(seedCutScan, num[0]);
  h_scan_seed_den_B->Fill(seedCutScan, den[0]);
  h_scan_seed_num_BtoC->Fill(seedCutScan, num[1]);
  h_scan_seed_den_BtoC->Fill(seedCutScan, den[1]);
  h_scan_seed_num_C->Fill(seedCutScan, num[2]);
  h_scan_seed_den_C->Fill(seedCutScan, den[2]);
  h_scan_seed_fake_B->Fill(seedCutScan, fake[0]);
  h_scan_seed_fake_BtoC->Fill(seedCutScan, fake[1]);
  h_scan_seed_fake_C->Fill(seedCutScan, fake[2]);
  h_scan_seed_events->Fill(seedCutScan, 1.0f);
}

for (float nodeCutScan : scanSVcuts) {
  auto compsNode = buildComponents(SeedSVcut, nodeCutScan, EdgeMin);
  auto [num, den, fake] = evaluateComps(compsNode, pSVcut, PVVetoCut);
  h_scan_node_num_B->Fill(nodeCutScan, num[0]);
  h_scan_node_den_B->Fill(nodeCutScan, den[0]);
  h_scan_node_num_BtoC->Fill(nodeCutScan, num[1]);
  h_scan_node_den_BtoC->Fill(nodeCutScan, den[1]);
  h_scan_node_num_C->Fill(nodeCutScan, num[2]);
  h_scan_node_den_C->Fill(nodeCutScan, den[2]);
  h_scan_node_fake_B->Fill(nodeCutScan, fake[0]);
  h_scan_node_fake_BtoC->Fill(nodeCutScan, fake[1]);
  h_scan_node_fake_C->Fill(nodeCutScan, fake[2]);
  h_scan_node_events->Fill(nodeCutScan, 1.0f);
}

for (float edgeCutScan : scanSVcuts) {
  auto compsEdge = buildComponents(SeedSVcut, NodeSVmin, edgeCutScan);
  auto [num, den, fake] = evaluateComps(compsEdge, pSVcut, PVVetoCut);
  h_scan_edge_num_B->Fill(edgeCutScan, num[0]);
  h_scan_edge_den_B->Fill(edgeCutScan, den[0]);
  h_scan_edge_num_BtoC->Fill(edgeCutScan, num[1]);
  h_scan_edge_den_BtoC->Fill(edgeCutScan, den[1]);
  h_scan_edge_num_C->Fill(edgeCutScan, num[2]);
  h_scan_edge_den_C->Fill(edgeCutScan, den[2]);
  h_scan_edge_fake_B->Fill(edgeCutScan, fake[0]);
  h_scan_edge_fake_BtoC->Fill(edgeCutScan, fake[1]);
  h_scan_edge_fake_C->Fill(edgeCutScan, fake[2]);
  h_scan_edge_events->Fill(edgeCutScan, 1.0f);
}


		
	
	//Clustering
	//

	std::vector<TracksClusteringFromDisplacedSeed::Cluster> clusters;
	clusters.reserve(comps.size());
	
	
	for (auto const& comp : comps) {
	

	  // pick seed node = strongest node for that cls
	  int seedIdx = comp[0];
	
	  TracksClusteringFromDisplacedSeed::Cluster aCl;
	  aCl.seedingTrack = t_trks_SV[seedIdx];
	  aCl.seedPoint    = GlobalPoint(pv.x(), pv.y(), pv.z());
	
	  aCl.tracks.clear();
	  aCl.tracks.reserve(comp.size());
	  for (int idx : comp) {
	    aCl.tracks.push_back(t_trks_SV[toTrkIdx(idx)]);
	  }
	
	  clusters.push_back(std::move(aCl));
	}

   	 std::cout << "clusters size" << clusters.size() << std::endl;

	

	
   	 // IVF default clustering

   	//std::vector<TracksClusteringFromDisplacedSeed::Cluster> clusters = clusterizer->clusters(pv, t_trks_SV);
   	
   	//MODEL based cluster filtering

   	   
   	//GENMATCHED based clustering
   	
   	//std::vector<TracksClusteringFromDisplacedSeed::Cluster> clustersAll =
   	// clusterizer->clusters(pv, t_trks_SV);

   	//std::vector<TracksClusteringFromDisplacedSeed::Cluster> clusters;
   	////
   	//// helper lambda: match track index by (pt, eta, phi) within tolerance
   	//auto matchIndex = [&](const reco::TransientTrack& trk) -> int {
   	//  const auto& ref = trk.track();
   	//  for (size_t i = 0; i < t_trks_SV.size(); ++i) {
   	//    const auto& cand = t_trks_SV[i].track();
   	//    if (std::abs(cand.pt()  - ref.pt())  < 1e-5 &&
   	//        std::abs(cand.eta() - ref.eta()) < 1e-5 &&
   	//        std::abs(reco::deltaPhi(cand.phi(), ref.phi())) < 1e-5) {
   	//      return static_cast<int>(i);
   	//    }
   	//  }
   	//  return -1;  // not found
   	//};

   	////Keep based on seed track
   	//for (auto& cl : clustersAll) {
   	//  int k = matchIndex(cl.seedingTrack);
   	//  if (k >= 0) {
   	//      size_t origIdx = t_trks_SV_indices[k];
   	//      if (genmatched_indices.count(origIdx)) {
   	//          clusters.push_back(std::move(cl));
   	//      }
   	//  }
   	//}

   	//Keep based on number of tracks that are present in genmatched indices
   	//int nRequiredGenMatchedTracks = 1;  // <-- configurable

   	//for (auto& cl : clustersAll) {
   	//    int nGenMatched = 0;
   	//
   	//    // loop over all tracks in the cluster
   	//    for (const auto& trk : cl.tracks) {
   	//        int k = matchIndex(trk);
   	//        if (k >= 0) {
   	//            size_t origIdx = t_trks_SV_indices[k];
   	//            if (expanded_indices.count(origIdx)) { // check if gen matched
   	//                ++nGenMatched;
   	//                if (nGenMatched >= nRequiredGenMatchedTracks) break;
   	//            }
   	//        }
   	//    }
   	//
   	//    if (nGenMatched >= nRequiredGenMatchedTracks) {
   	//        clusters.push_back(std::move(cl));
   	//    }
   	//}

   	std::vector<TransientVertex> recoVertices;
   	VertexDistanceXY vertTool2D;
   	VertexDistance3D vertTool3D;
   	for (std::vector<TracksClusteringFromDisplacedSeed::Cluster>::iterator cluster = clusters.begin(); cluster != clusters.end(); ++cluster) {
   	       if (cluster->tracks.size()<2) continue; 
   	       
   	       // fit one or more vertices from the tracks in one cluster
   	       std::vector<TransientVertex> tmp_vertices = vtxmaker_.vertices(cluster->tracks);
   	       
   	       
   	       
   	       for (std::vector<TransientVertex>::iterator v = tmp_vertices.begin(); v != tmp_vertices.end(); ++v) {

   	         reco::Vertex tmpvtx(*v);
   	         // Compute flight distance significance w.r.t. primary vertex
   	         // Assume primary vertex is the first vertex in recoVertices
   	         //if (!recoVertices.empty()) {
   	         //    Measurement1D dist2D = vertTool2D.distance(tmpvtx, pv);
   	         //    Measurement1D dist3D = vertTool3D.distance(tmpvtx, pv);

   	         //    // Apply your significance cut
   	         //    if (dist2D.significance() < 2.5 || dist3D.significance() < 0.5) {
   	         //        continue; // skip this vertex
   	         //    }
   	         //}

   	         
   	             recoVertices.push_back(*v);
   	       }
   	 }

	std::cout << "Vertex size after AVR: " << recoVertices.size() << std::endl;


   	 vertexMerge(recoVertices, 0.7, 2);

	std::cout << "Vertex size after merge1: " << recoVertices.size() << std::endl;


   	 double dRCut              = 1000.0;
   	 double distCut            = 1000.0;
   	 double sigCut             = 1000.0; 
   	 double dLenFraction       = 1.0; 
   	 double fitterSigmacut     = 3.0; 
   	 double fitterTini         = 256.0; 
   	 double fitterRatio        = 0.25; 
   	 double maxTimeSig         = 9999.0;
   	 int    trackMinLayers     = 2;
   	 double trackMinPt         = 0.4;
   	 int    trackMinPixels     = 1;
   	     
   	 std::vector<TransientVertex> newVTXs = TrackVertexArbitrator(
   	   pv,
   	   beamSpotHandle,
   	   recoVertices,
   	   t_trks_SV,
   	   dRCut,
   	   distCut,
   	   sigCut,
   	   dLenFraction,
   	   fitterSigmacut,
   	   fitterTini,
   	   fitterRatio,
   	   maxTimeSig,
   	   trackMinLayers,
   	   trackMinPt,
   	   trackMinPixels
   	);
   	//std::vector<TransientVertex> newVTXs = recoVertices;

	std::cout << "Vertex size after arbitration: " << newVTXs.size() << std::endl;

  
   	vertexMerge(newVTXs, 0.2, 10);
   	std::cout << "newVTXs size" << newVTXs.size() << std::endl;


   	int nvtx=0;
   	for(size_t ivtx=0; ivtx<newVTXs.size(); ivtx++){
   		    reco::Vertex tmpvtx(newVTXs[ivtx]);
   	     const auto& vtx = newVTXs[ivtx];
   	     const auto& trks = vtx.originalTracks();
   	     int track_per_sv_counter = 0;
   	     for (const auto& ttrk : trks) {
   	         track_per_sv_counter++;
   	             //const reco::Track& trk = ttrk.track();
   	         //reco::CandidatePtr trackPtr = 
   	         SVrecoTrk_SVrecoIdx.push_back(nvtx);
   	         const reco::Track& trk = ttrk.track();
   	         SVrecoTrk_pt.push_back(trk.pt());
   	         SVrecoTrk_eta.push_back(trk.eta());
   	         SVrecoTrk_phi.push_back(trk.phi());

   	         //reco::TransientTrack ttrk = theB->build(trk);
   	         auto ip3d_trk = IPTools::signedImpactParameter3D(ttrk, direction, pv).second;
   	         auto ip2d_trk = IPTools::signedTransverseImpactParameter(ttrk, direction, pv).second;
   	         SVrecoTrk_ip3d.push_back(ip3d_trk.value());
   	         SVrecoTrk_ip2d.push_back(ip2d_trk.value());




   	     }
   	     nvtx++;
   	     SV_x_reco.push_back(tmpvtx.position().x());
   	     SV_y_reco.push_back(tmpvtx.position().y());
   	     SV_z_reco.push_back(tmpvtx.position().z());
   	     SV_reco_nTracks.push_back(track_per_sv_counter);
   	     SV_chi2_reco.push_back(tmpvtx.normalizedChi2()); 
   	}
   	nSVs_reco.push_back(nvtx);
    }

    
   int nhads = 0;
   int ngv = 0;
   int ngv_tau = 0;
   int ngv_b = 0;
   int ngv_s = 0;
   int ngv_d = 0;
   std::vector<float> temp_Daughters_pt;
   //std::vector<float> temp_Daughters_pdg;
   std::vector<float> temp_Daughters_eta;
   std::vector<float> temp_Daughters_phi;
   std::vector<int> temp_Daughters_charge;
   std::vector<int> temp_Daughters_hadidx;
   std::vector<int> temp_Daughters_flav;
   std::vector<int> temp_Daughters_label;

   static const std::unordered_set<int> pdgSet_B = {
        521, 511, 531, 541,        // Bottom mesons
        5122,                      // Lambda_b0
        5132, 5232, 5332,          // Xi_b-, Xi_b0, Omega_b-
        5142, 5242, 5342,          // Xi_bc0, Xi_bc+, Omega_bc0
        5512, 5532, 5542, 5554     // Xi_bb-, Omega_bb-, Omega_bbc0, Omega_bbb-
    };

    static const std::unordered_set<int> pdgSet_D = {
        411, 421, 431,             // Charmed mesons
        4122,                      // Lambda_c
        4232, 4132, 4332,          // Xi_c+, Xi’c0, Omega_c0
        4412, 4422, 4432, 4444     // Xi_cc+, Xi_cc++, Omega_cc+, Omega_ccc++
    };

    static const std::unordered_set<int> pdgSet_S = {
        3122, 3222, 3212,          // Lambda, Sigma+, Sigma0
        3312, 3322, 3334           // Xi-, Xi0, Omega-
    };

    static const std::unordered_set<int> pdgSet_Tau = {15};

   for (size_t i = 0; i < merged->size(); ++i) 
   {//Prune loop
       temp_Daughters_pt.clear();
       temp_Daughters_eta.clear();
       temp_Daughters_phi.clear();
       temp_Daughters_charge.clear();
       temp_Daughters_hadidx.clear();
       temp_Daughters_label.clear();
       temp_Daughters_flav.clear();

       const reco::Candidate* hadron = &(*merged)[i];
       if (!(hadron->pt() > 0.0 && std::abs(hadron->eta()) < 2.5)) continue;

       int hadPDG = checkPDG(std::abs(hadron->pdgId())); // 1=B, 2=D, 3=S, 4=tau, 5=other

       int nStableCharged = 0;
       float vx = 0;
       float vy = 0;
       float vz = 0;

       // Keep track if  stored GV for this hadron
       bool addedGV = false;

       // ----------------------------------------
       // Loop over all potential daughters
       // ----------------------------------------
       for (size_t k = 0; k < merged->size(); ++k) {
           const reco::GenParticle* dau = &(*merged)[k];
           if (dau->status() != 1) continue;
           if (dau->charge() == 0) continue;
           if (dau->pt() < 0.8) continue;
           if (std::abs(dau->eta()) > 2.5) continue;

           // Trace ancestry to see if 'hadron' is an ancestor
           bool isDescendant = false;
           const reco::Candidate* mom = dau->mother(0);
           while (mom) {
               if (mom == hadron) { isDescendant = true; break; }

               int mPDG = std::abs(mom->pdgId());
               if (pdgSet_B.count(mPDG) || pdgSet_D.count(mPDG) ||
                   pdgSet_S.count(mPDG) || pdgSet_Tau.count(mPDG))
                   break;
               mom = mom->mother(0);
           }
           if (!isDescendant) continue;

           // --- Classify and store daughter ---
           int label = getDaughterLabel(dau);
           nStableCharged++;

           temp_Daughters_pt.push_back(dau->pt());
           temp_Daughters_eta.push_back(dau->eta());
           temp_Daughters_phi.push_back(dau->phi());
           temp_Daughters_charge.push_back(dau->charge());
           temp_Daughters_hadidx.push_back(ngv);     // group ID for hadron
           temp_Daughters_flav.push_back(hadPDG);  // hadron flavor (5 if light)
           temp_Daughters_label.push_back(label);  // classification label (0–5)

           // Use daughter vertex as approximate GV
           vx = dau->vx();
           vy = dau->vy();
           vz = dau->vz();
       }

       // Require atleast 2 stable charged daughters
       if (nStableCharged < 2) continue;

       // -----------------------------------------
       // Store GV + hadron info once
       // -----------------------------------------
       if (!addedGV) {
           nhads++;
           ngv++; // increment hadron group ID
           addedGV = true;

           if (hadPDG == 1) ngv_b++;
           if (hadPDG == 2) ngv_d++;
           if (hadPDG == 3) ngv_s++;
           if (hadPDG == 4) ngv_tau++;

           Hadron_pt.push_back(hadron->pt());
           Hadron_eta.push_back(hadron->eta());
           Hadron_phi.push_back(hadron->phi());
           Hadron_GVx.push_back(vx);
           Hadron_GVy.push_back(vy);
           Hadron_GVz.push_back(vz);
           GV_flag.push_back(nhads - 1);
       }

       // Append daughters (only after GV stored)
       Daughters_pt.insert(Daughters_pt.end(), temp_Daughters_pt.begin(), temp_Daughters_pt.end());
       Daughters_eta.insert(Daughters_eta.end(), temp_Daughters_eta.begin(), temp_Daughters_eta.end());
       Daughters_phi.insert(Daughters_phi.end(), temp_Daughters_phi.begin(), temp_Daughters_phi.end());
       Daughters_charge.insert(Daughters_charge.end(), temp_Daughters_charge.begin(), temp_Daughters_charge.end());
       Daughters_hadidx.insert(Daughters_hadidx.end(), temp_Daughters_hadidx.begin(), temp_Daughters_hadidx.end());
       Daughters_flav.insert(Daughters_flav.end(), temp_Daughters_flav.begin(), temp_Daughters_flav.end());
       Daughters_label.insert(Daughters_label.end(), temp_Daughters_label.begin(), temp_Daughters_label.end());

       nDaughters.push_back(nStableCharged);
       nDaughters_B.push_back(hadPDG == 1 ? nStableCharged : 0);
       nDaughters_D.push_back(hadPDG == 2 ? nStableCharged : 0);
    } //prune loop

  
   nHadrons.push_back(nhads);
   nGV.push_back(ngv);
   nGV_B.push_back(ngv_b);
   nGV_S.push_back(ngv_s);
   nGV_D.push_back(ngv_d);
   nGV_Tau.push_back(ngv_tau);


   int nsvs = 0;
   for(const auto &sv: *svHandle){
	nsvs++;
   	SV_x.push_back(sv.vertex().x());
	SV_y.push_back(sv.vertex().y());
	SV_z.push_back(sv.vertex().z());
	SV_pt.push_back(sv.pt());
	SV_mass.push_back(sv.p4().M());
	SV_ntrks.push_back(sv.numberOfSourceCandidatePtrs());

	for(size_t i =0; i < sv.numberOfSourceCandidatePtrs(); ++i){
	    reco::CandidatePtr trackPtr = sv.sourceCandidatePtr(i);

	    SVtrk_pt.push_back(trackPtr->pt());
	    SVtrk_eta.push_back(trackPtr->eta());
	    SVtrk_phi.push_back(trackPtr->phi());
        SVtrk_SVIdx.push_back(nsvs-1);
        const pat::PackedCandidate* mypacked = dynamic_cast<const pat::PackedCandidate*>(trackPtr.get());
        if (mypacked && mypacked->hasTrackDetails()) {
        const reco::Track& trk = mypacked->pseudoTrack();

        // dz and dz significance
        double dz    = trk.dz(pv.position());
        double dzErr = trk.dzError();
        SVtrk_ipz.push_back(dz);
        SVtrk_ipzsig.push_back(dzErr > 0 ? dz / dzErr : 0);

        // dxy and dxy significance
        double dxy    = trk.dxy(pv.position());
        double dxyErr = trk.dxyError();
        SVtrk_ipxy.push_back(dxy);
        SVtrk_ipxysig.push_back(dxyErr > 0 ? dxy / dxyErr : 0);

        // 3D IP and significance
        reco::TransientTrack ttrk = theB->build(trk);
        auto ip3dPair = IPTools::absoluteImpactParameter3D(ttrk, pv);
        if (ip3dPair.first) {
            SVtrk_ip3d.push_back(ip3dPair.second.value());
            SVtrk_ip3dsig.push_back(ip3dPair.second.significance());
        } else {
            SVtrk_ip3d.push_back(-999);
            SVtrk_ip3dsig.push_back(-999);
        }
    } else {
        // No track details — fill dummy values
        SVtrk_ipz.push_back(-999);
        SVtrk_ipzsig.push_back(-999);
        SVtrk_ipxy.push_back(-999);
        SVtrk_ipxysig.push_back(-999);
        SVtrk_ip3d.push_back(-999);
        SVtrk_ip3dsig.push_back(-999);
    }
	}


   }
   nSVs.push_back(nsvs);
   std::cout<<"Vertices from IVF : "<< nsvs<<std::endl;
   

   // Here perform the matching
   //distances[i][j] = 3D Euclidean distance between:
    //SV i: position (SV_x[i], SV_y[i], SV_z[i])
    //Hadron j: position (Hadron_GVx[j], Hadron_GVy[j], Hadron_GVz[j])

    auto distances = computeDistanceMatrix(SV_x, SV_y, SV_z, Hadron_GVx, Hadron_GVy, Hadron_GVz);
    
    //distances[i][j] is the distance bewteen SV_i and GV_j
    //std::cout<<"Distance SV 0 GV 0 is "<<distances[0][0]<<std::endl;
    //printDistanceMatrix(distances);
    
    auto result = matchHadronsToSV(
                                        distances,
                                        SVtrk_pt, SVtrk_eta, SVtrk_phi, SVtrk_SVIdx,
                                        Daughters_pt, Daughters_eta, Daughters_phi, Daughters_hadidx,
                                        Hadron_GVx.size()
                                        );
    Hadron_SVIdx = result.first;
    Hadron_SVDistance = result.second;



    auto distances_reco = computeDistanceMatrix(SV_x_reco, SV_y_reco, SV_z_reco, Hadron_GVx, Hadron_GVy, Hadron_GVz);
    //distances[i][j] is the distance bewteen SV_i and GV_j
    //std::cout<<"Distance SV 0 GV 0 is "<<distances[0][0]<<std::endl;
    //printDistanceMatrix(distances);
    //std::cout<<"Distances are computed"<<std::endl;
    auto result_reco = matchHadronsToSV(
                                        distances_reco,
                                        SVrecoTrk_pt, SVrecoTrk_eta, SVrecoTrk_phi, SVrecoTrk_SVrecoIdx,
                                        Daughters_pt, Daughters_eta, Daughters_phi, Daughters_hadidx,
                                        Hadron_GVx.size()
                                        );
    Hadron_SVRecoIdx = result_reco.first;
    Hadron_SVRecoDistance = result_reco.second;












   tree->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void DemoAnalyzer::beginStream(edm::StreamID) {

	tree->Branch("run", &run_, "run/i");
   	tree->Branch("lumi", &lumi_, "lumi/i");
   	tree->Branch("evt", &evt_);
	tree->Branch("nPU", &nPU);
	tree->Branch("nHadrons", &nHadrons);                // Hadrons of GenVertices             
	tree->Branch("Hadron_pt", &Hadron_pt);              // Hadrons of GenVertices
    	tree->Branch("Hadron_SVIdx", &Hadron_SVIdx);              // Hadrons of GenVertices
    	tree->Branch("Hadron_SVDistance", &Hadron_SVDistance);
	tree->Branch("Hadron_eta", &Hadron_eta);            // Hadrons of GenVertices 
	tree->Branch("Hadron_phi", &Hadron_phi);            // Hadrons of GenVertices 
	tree->Branch("Hadron_GVx", &Hadron_GVx);            // Hadrons of GenVertices 
	tree->Branch("Hadron_GVy", &Hadron_GVy);            // Hadrons of GenVertices 
	tree->Branch("Hadron_GVz", &Hadron_GVz);            // Hadrons of GenVertices 
	tree->Branch("nGV", &nGV);
	tree->Branch("nGV_B", &nGV_B);
    	tree->Branch("nGV_Tau", &nGV_Tau);
    	tree->Branch("nGV_S", &nGV_S);
	tree->Branch("nGV_D", &nGV_D);
	tree->Branch("GV_flag", &GV_flag);                  // GV_genHadronIdx 
	tree->Branch("nDaughters", &nDaughters);
	tree->Branch("nDaughters_B", &nDaughters_B);
	tree->Branch("nDaughters_D", &nDaughters_D);
	tree->Branch("Daughters_hadidx", &Daughters_hadidx);
    	tree->Branch("Daughters_flav", &Daughters_flav);        // flavor of the mother of the vertex
	tree->Branch("Daughters_label", &Daughters_label);
	tree->Branch("Daughters_pt", &Daughters_pt);        // from 0.8 on
	tree->Branch("Daughters_eta", &Daughters_eta);
	tree->Branch("Daughters_phi", &Daughters_phi);      // ok
	tree->Branch("Daughters_charge", &Daughters_charge); // ok +-1
	
	tree->Branch("nTrks", &ntrks);                      // all the tracks in the event
	tree->Branch("trk_ip2d", &trk_ip2d);
	tree->Branch("trk_ip3d", &trk_ip3d);
    	tree->Branch("trk_dz", &trk_ipz);
    	tree->Branch("trk_dzsig", &trk_ipzsig);
	tree->Branch("trk_ip2dsig", &trk_ip2dsig);
    	tree->Branch("trk_ip3dsig", &trk_ip3dsig);
	tree->Branch("trk_p", &trk_p);
	tree->Branch("trk_pt", &trk_pt);
	tree->Branch("trk_eta", &trk_eta);
	tree->Branch("trk_phi", &trk_phi);
	tree->Branch("trk_nValid", &trk_nValid);
	tree->Branch("trk_nValidPixel", &trk_nValidPixel);
	tree->Branch("trk_nValidStrip", &trk_nValidStrip);
	tree->Branch("trk_charge", &trk_charge);
         
    	tree->Branch("trk_i", &trk_i);                  // needed for edge features
	tree->Branch("trk_j", &trk_j);                  // needed for edge features
	tree->Branch("deltaR", &deltaR);                
	tree->Branch("dca", &dca);
	tree->Branch("dca_sig", &dca_sig);
    	tree->Branch("cptopv", &cptopv);
	tree->Branch("pvtoPCA_i", &pvtoPCA_i);
	tree->Branch("pvtoPCA_j", &pvtoPCA_j);
	tree->Branch("dotprod_i", &dotprod_i);
	tree->Branch("dotprod_j", &dotprod_j);
	tree->Branch("pair_mom", &pair_mom);
	tree->Branch("pair_invmass", &pair_invmass);    // inv mass


	//tree->Branch("nJets", &njets);
	//tree->Branch("jet_pt", &jet_pt);
    //tree->Branch("jet_eta", &jet_eta);
    //tree->Branch("jet_phi", &jet_phi);

	tree->Branch("nSVs", &nSVs);
	tree->Branch("SV_x", &SV_x);
	tree->Branch("SV_y", &SV_y);
	tree->Branch("SV_z", &SV_z);
	tree->Branch("SV_pt", &SV_pt);
	tree->Branch("SV_mass", &SV_mass);
	tree->Branch("SV_ntrks", &SV_ntrks);
	tree->Branch("SVtrk_pt", &SVtrk_pt);
    tree->Branch("SVtrk_SVIdx", &SVtrk_SVIdx);
	tree->Branch("SVtrk_eta", &SVtrk_eta);
	tree->Branch("SVtrk_phi", &SVtrk_phi);
    tree->Branch("SVtrk_ipz", &SVtrk_ipz);
    tree->Branch("SVtrk_ipzsig", &SVtrk_ipzsig);
    tree->Branch("SVtrk_ipxy", &SVtrk_ipxy);
    tree->Branch("SVtrk_ipxysig", &SVtrk_ipxysig);
    tree->Branch("SVtrk_ip3d", &SVtrk_ip3d);
    tree->Branch("SVtrk_ip3dsig", &SVtrk_ip3dsig);
        
    tree->Branch("Edge_probs", &preds);
	tree->Branch("cut_val", &cut);

	tree->Branch("nSVs_reco", &nSVs_reco);
    tree->Branch("SV_x_reco", &SV_x_reco);
    tree->Branch("Hadron_SVRecoIdx", &Hadron_SVRecoIdx);              // Hadrons of GenVertices
    tree->Branch("Hadron_SVRecoDistance", &Hadron_SVRecoDistance);
    tree->Branch("SVrecoTrk_SVrecoIdx", &SVrecoTrk_SVrecoIdx);
    tree->Branch("SVrecoTrk_pt", &SVrecoTrk_pt);
    tree->Branch("SVrecoTrk_eta", &SVrecoTrk_eta);
    tree->Branch("SVrecoTrk_phi", &SVrecoTrk_phi);
    tree->Branch("SVrecoTrk_ip3d", &SVrecoTrk_ip3d);
    tree->Branch("SV_y_reco", &SV_y_reco);
    tree->Branch("SV_z_reco", &SV_z_reco);
    tree->Branch("SV_reco_nTracks", &SV_reco_nTracks);
    tree->Branch("SV_chi2_reco", &SV_chi2_reco);

    tree->Branch("SV_score_sig", &sv_score_sig);
    tree->Branch("SV_score_bkg", &sv_score_bkg);
    tree->Branch("Edge_score_sig", &edge_score_sig);
    tree->Branch("Edge_score_bkg", &edge_score_bkg);

    tree->Branch("nHad_B", &nHad_B);
    tree->Branch("nHad_BtoC", &nHad_BtoC);
    tree->Branch("nHad_C", &nHad_C);
    tree->Branch("eff_B", &eff_B);
    tree->Branch("eff_BtoC", &eff_BtoC);
    tree->Branch("eff_C", &eff_C);
    tree->Branch("fake_B", &fake_B);
    tree->Branch("fake_BtoC", &fake_BtoC);
    tree->Branch("fake_C", &fake_C);
    

	auto strip_chars = [&](std::string& s, const std::string& chars) {
	  s.erase(std::remove_if(s.begin(), s.end(),
	                         [&](char c){ return chars.find(c) != std::string::npos; }),
	          s.end());
	};
	
	auto trim_inplace = [&](std::string& s) {
	  auto notspace = [](unsigned char c){ return !std::isspace(c); };
	  s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
	  s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
	};
	
	auto parse_int_list = [&](std::string s) -> std::vector<int> {
	  strip_chars(s, "\"[]");
	  std::vector<int> out;
	  std::stringstream ss(s);
	  std::string tok;
	  while (std::getline(ss, tok, ',')) {
	    trim_inplace(tok);
	    if (!tok.empty()) out.push_back(std::stoi(tok));
	  }
	  return out;
	};

	auto splitCSVQuoted = [&](const std::string& line,
                          std::vector<std::string>& fields,
                          size_t expected) -> bool {
	  fields.clear();
	  std::string cur;
	  bool inQuotes = false;
	
	  for (size_t i = 0; i < line.size(); ++i) {
	    char c = line[i];
	
	    if (c == '"') {
	      // handle escaped quote ""
	      if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
	        cur.push_back('"');
	        ++i;
	      } else {
	        inQuotes = !inQuotes;
	      }
	    } else if (c == ',' && !inQuotes) {
	      fields.push_back(cur);
	      cur.clear();
	    } else {
	      cur.push_back(c);
	    }
	  }
	  fields.push_back(cur);
	
	  return (fields.size() == expected);
	};
    

	std::ifstream file(genmatch_csv_);
	std::string line;
	unsigned int line_no = 0;
	std::vector<std::string> fields;
	
	while (std::getline(file, line)) {
	    ++line_no;
	    if (line.empty()) continue;
	  
	    if (!splitCSVQuoted(line, fields, 5)) {
	      std::cerr << "Line " << line_no
	                << ": bad CSV field count = " << fields.size()
	                << "\nLine: " << line << "\n";
	      continue;
	    }
	  
	    const std::string& run_str    = fields[0];
	    const std::string& lumi_str   = fields[1];
	    const std::string& evt_str    = fields[2];
	    const std::string& labels_str = fields[3];
	    const std::string& hadidx_str = fields[4];
	  
	    if (run_str == "run") continue; // header
	  
	    const unsigned int run  = std::stoul(run_str);
	    const unsigned int lumi = std::stoul(lumi_str);
	    const unsigned int evt  = std::stoul(evt_str);
	  
	    const std::vector<int> trk_labels = parse_int_list(labels_str);
	    const std::vector<int> trk_hadidx = parse_int_list(hadidx_str);
	  
	    if (trk_labels.size() != trk_hadidx.size()) {
	      std::cerr << "Line " << line_no
	                << ": size mismatch labels=" << trk_labels.size()
	                << " hadidx=" << trk_hadidx.size() << "\n";
	      continue;
	    }
 
	
	    SigMatchEntry entry;
	    entry.indices.reserve(trk_labels.size());
	    entry.labels.reserve(trk_labels.size());
	    entry.hadidx.reserve(trk_labels.size());
	
	    for (size_t i = 0; i < trk_labels.size(); ++i) {
	      const int lab = trk_labels[i];
	      if (lab == 2 || lab == 3 || lab == 4) {
	        entry.indices.push_back(static_cast<int>(i));  // track index
	        entry.labels.push_back(lab);                   // label at i
	        entry.hadidx.push_back(trk_hadidx[i]);         // hadidx at i
	      }
	    }
	
	    sigMatchMap_[{run, lumi, evt}] = std::move(entry);
	
	  } 
   	
}


// ------------ method called once each job just after ending the event loop  ------------
//void DemoAnalyzer::endJob() {
//}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DemoAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault( desc);
  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DemoAnalyzer);
