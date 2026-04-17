#ifndef DemoAnalyzer_h
#define DemoAnalyzer_h

// system include files
#include <memory>
#include <tuple>
#include <optional>
#include <limits>
#include <cmath>
#include <algorithm>
#include <utility>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexFitter.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"
#include "RecoVertex/AdaptiveVertexFinder/interface/TracksClusteringFromDisplacedSeed.h"
#include "RecoVertex/AdaptiveVertexFinder/interface/VertexMerging.h"
#include "RecoVertex/AdaptiveVertexFinder/interface/TrackVertexArbitration.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/VertexTools/interface/SharedTracks.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexSmoother.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "RecoVertex/AdaptiveVertexFinder/interface/SVTimeHelpers.h"

//TFile Service

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//ROOT
#include "TTree.h"
#include "TVector3.h"
#include "math.h"

//Transient Track
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"

//Pileup info
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

//IPTOOLS
#include "TrackingTools/IPTools/interface/IPTools.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/Math/interface/deltaPhi.h"

//ONNX
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
//
// class declaration
//
using namespace cms::Ort;

class DemoAnalyzer : public edm::stream::EDAnalyzer<edm::GlobalCache<ONNXRuntime>> {
   public:
      explicit DemoAnalyzer (const edm::ParameterSet&, const ONNXRuntime *);
      ~DemoAnalyzer();
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
      static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet &);
      static void globalEndJob(const ONNXRuntime *);
      
   private:
      //virtual void beginJob() override;
      virtual void beginStream(edm::StreamID) override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      //virtual void endJob() override;
      int checkPDG(int abs_pdg);
      int getDaughterLabel(const reco::GenParticle *);
      bool isGoodVtx(TransientVertex &);
      std::vector<TransientVertex> TrackVertexRefit(std::vector<reco::TransientTrack> &, std::vector<TransientVertex> &);
      void vertexMerge(std::vector<TransientVertex> &, double, double );
      std::vector<TransientVertex> TrackVertexArbitrator(const reco::Vertex&, const edm::Handle<reco::BeamSpot>&, const::std::vector<TransientVertex>&, std::vector<reco::TransientTrack>&, double, double, double, double, double, double, double, double, int, double, int);
      float sigmoid(float x);
      
      const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> theTTBToken;
      bool training_;
      edm::EDGetTokenT<pat::PackedCandidateCollection> TrackCollT_;
      edm::EDGetTokenT<reco::VertexCollection> PVCollT_;
      edm::EDGetTokenT<edm::View<reco::VertexCompositePtrCandidate>> SVCollT_;
      edm::EDGetTokenT<pat::PackedCandidateCollection> LostTrackCollT_;
      edm::EDGetTokenT<edm::View<reco::Jet> > jet_collT_;
      edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;
      edm::EDGetTokenT<edm::View<reco::GenParticle> > prunedGenToken_;
      edm::EDGetTokenT<edm::View<pat::PackedGenParticle> > packedGenToken_;
      edm::EDGetTokenT<edm::View<reco::GenParticle> > mergedGenToken_;
      double TrackPtCut_;
      double TrackPredCut_;
      edm::ParameterSet vtxconfig_;
      ConfigurableVertexReconstructor vtxmaker_;
      edm::EDGetTokenT<std::vector<PileupSummaryInfo>> PupInfoT_;
      double vtxweight_;
      std::unique_ptr<TracksClusteringFromDisplacedSeed> clusterizer;
      std::string genmatch_csv_;

      struct SigMatchEntry {
      std::vector<int> indices;  // track indices i
      std::vector<int> labels;   // trk_labels[i]
      std::vector<int> hadidx;   // trk_hadidx[i]
      };

      std::map<std::tuple<unsigned int, unsigned int, unsigned int>, SigMatchEntry> sigMatchMap_;


      unsigned int run_;
      unsigned int lumi_;
      unsigned int evt_;

      TTree *tree;
      float nPU;

      std::vector<float> Hadron_pt;
      std::vector<float> Hadron_eta;
      std::vector<float> Hadron_phi;
      std::vector<float> Hadron_GVx;
      std::vector<float> Hadron_GVy;
      std::vector<float> Hadron_GVz;
      std::vector<int> Hadron_SVIdx;
      std::vector<float> Hadron_SVDistance;
      
      std::vector<int> nHadrons;
      std::vector<int> nGV;
      std::vector<int> nGV_B;
      std::vector<int> nGV_Tau;
      std::vector<int> nGV_S;
      std::vector<int> nGV_D;
      std::vector<int> GV_flag;
      std::vector<int> nDaughters;
      std::vector<int> nDaughters_B;
      std::vector<int> nDaughters_S;
      std::vector<int> nDaughters_D;
      std::vector<int> Daughters_hadidx;
      std::vector<int> Daughters_flav;
      std::vector<int> Daughters_label;
      std::vector<float> Daughters_pt;
      std::vector<float> Daughters_eta;
      std::vector<float> Daughters_phi;
      std::vector<float> Daughters_charge;
           
      std::vector<int> ntrks;
      std::vector<float> trk_ip2d;
      std::vector<float> trk_ip3d;
      std::vector<float> trk_ipz;
      std::vector<float> trk_ipzsig;
      std::vector<float> trk_ip2dsig;
      std::vector<float> trk_ip3dsig;
      std::vector<float> trk_p;
      std::vector<float> trk_pt;
      std::vector<float> trk_eta;
      std::vector<float> trk_phi;
      std::vector<float> trk_charge;
      std::vector<int> trk_nValid;
      std::vector<int> trk_nValidPixel;
      std::vector<int> trk_nValidStrip;
	
      std::vector<int> trk_i;
      std::vector<int> trk_j;
      std::vector<float> deltaR;
      std::vector<float> dca;
      std::vector<float> dca_sig;
      std::vector<float> cptopv;
      std::vector<float> pvtoPCA_i;
      std::vector<float> pvtoPCA_j;
      std::vector<float> dotprod_i;
      std::vector<float> dotprod_j;
      std::vector<float> pair_mom;
      std::vector<float> pair_invmass;
      
      
      std::vector<int> njets;
      std::vector<float> jet_pt;
      std::vector<float> jet_eta;
      std::vector<float> jet_phi;

      std::vector<int> nSVs;
      std::vector<float> SV_x;
      std::vector<float> SV_y;
      std::vector<float> SV_z;
      std::vector<float> SV_pt;
      std::vector<float> SV_mass;
      std::vector<int> SV_ntrks;
      std::vector<float> SVtrk_pt;
      std::vector<int> SVtrk_SVIdx;
      std::vector<float> SVtrk_eta;
      std::vector<float> SVtrk_phi; 
      std::vector<float> SVtrk_ipz; 
      std::vector<float> SVtrk_ipzsig; 
      std::vector<float> SVtrk_ipxy; 
      std::vector<float> SVtrk_ipxysig; 
      std::vector<float> SVtrk_ip3d; 
      std::vector<float> SVtrk_ip3dsig; 
      std::vector<float> SVrecoTrk_ip3d;
      std::vector<float> SVrecoTrk_ip2d;
      
      std::vector<float> preds;
      std::vector<float> cut;

      std::vector<int> nSVs_reco;
      std::vector<float> SV_x_reco;
      std::vector<float> SV_y_reco;
      std::vector<float> SV_z_reco;
      std::vector<int> SV_reco_nTracks;
      std::vector<float> SV_chi2_reco;
      std::vector<int> SVrecoTrk_SVrecoIdx;
      std::vector<float> SVrecoTrk_pt;
      std::vector<float> SVrecoTrk_eta;
      std::vector<float> SVrecoTrk_phi;
      std::vector<int> Hadron_SVRecoIdx;
      std::vector<float> Hadron_SVRecoDistance;

      std::vector<double> sv_score_sig;
      std::vector<double> sv_score_bkg;
      std::vector<double> edge_score_sig;
      std::vector<double> edge_score_bkg;

      std::vector<int> nHad_B;
      std::vector<int> nHad_BtoC;
      std::vector<int> nHad_C;
      std::vector<float> eff_B;
      std::vector<float> eff_BtoC;
      std::vector<float> eff_C;
      std::vector<int> fake_B;
      std::vector<int> fake_BtoC;
      std::vector<int> fake_C;


     
};

#endif // DemoAnalyzer_h

