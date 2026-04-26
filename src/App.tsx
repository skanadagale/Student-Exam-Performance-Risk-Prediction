import React, { useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  TrendingUp, 
  Users, 
  AlertCircle, 
  CheckCircle2, 
  BarChart3, 
  BrainCircuit, 
  UserPlus,
  RefreshCcw,
  BookOpen,
  Calendar,
  History,
  Info
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell,
  PieChart,
  Pie,
  ScatterChart,
  Scatter,
  ZAxis
} from 'recharts';
import { fetchAndProcessData, encodeData, splitData } from './services/dataService';
import { StudentRiskModel } from './services/mlService';
import { ProcessedStudent, ModelMetrics } from './types';
import { cn } from './lib/utils';
import { GoogleGenAI } from "@google/genai";

const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });

// --- Components ---

const StatCard = ({ title, value, icon: Icon, color }: { title: string, value: string | number, icon: any, color: string }) => (
  <motion.div 
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    className="bg-white/40 p-6 border-b border-black/10 flex flex-col justify-between h-32"
  >
    <div className="flex justify-between items-start">
      <span className="text-[10px] font-bold tracking-[0.2em] uppercase opacity-60">{title}</span>
      <Icon className={cn("w-4 h-4 opacity-30")} />
    </div>
    <p className="text-3xl font-serif italic text-slate-900">{value}</p>
  </motion.div>
);

const PredictionResult = ({ result, features, onReset }: { result: { prediction: number, probability: number }, features: any, onReset: () => void }) => {
  const [explanation, setExplanation] = useState<string>("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function getExplanation() {
      setLoading(true);
      try {
        const prompt = `A student has been predicted as ${result.prediction === 1 ? 'AT RISK of failing' : 'LIKELY TO PASS'} an exam.
        Data: Study ${features.studytime}, Absences ${features.absences}, Past Failures ${features.failures}, G1/G2: ${features.G1}/${features.G2}.
        Provide a 2-3 sentence empathetic expert analysis of risk factors.`;

        const response = await genAI.models.generateContent({
          model: "gemini-3-flash-preview",
          contents: prompt,
        });
        setExplanation(response.text || "Analysis complete.");
      } catch (err) {
        setExplanation("Analysis unavailable at the moment.");
      } finally {
        setLoading(false);
      }
    }
    getExplanation();
  }, [result, features]);

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex flex-col items-center justify-center text-center py-12 px-6"
    >
      <div className={cn(
        "mb-6 px-4 py-1 border text-[10px] font-bold tracking-widest uppercase",
        result.prediction === 1 ? "border-red-600 text-red-600" : "border-emerald-600 text-emerald-600"
      )}>
        Classification Result
      </div>
      
      <h1 className="font-serif text-[80px] md:text-[110px] leading-none mb-4 italic">
        {result.prediction === 1 ? "At Risk" : "Likely Pass"}
      </h1>
      
      <p className="max-w-md text-[14px] leading-relaxed opacity-80 mb-8">
        Based on the current configuration of study habits and previous performance metrics, this student has an <span className="font-bold">{(result.probability * 100).toFixed(1)}% probability</span> of reaching the targeted fail threshold.
      </p>

      <div className="w-full max-w-sm h-[1px] bg-black/10 relative mb-12">
        <motion.div 
          initial={{ left: 0 }}
          animate={{ left: `${(result.probability * 100)}%` }}
          className={cn("absolute -top-1 w-2 h-2 rounded-full", result.prediction === 1 ? "bg-red-600" : "bg-emerald-600")}
        />
        <div className="absolute left-0 -bottom-6 text-[9px] opacity-50 uppercase tracking-widest">Safe Level</div>
        <div className="absolute right-0 -bottom-6 text-[9px] opacity-50 uppercase tracking-widest">Critical High</div>
      </div>

      <div className="max-w-lg font-serif italic text-lg text-slate-700 bg-white/50 p-8 border-y border-black/10 mb-8">
        {loading ? "Authenticating Analysis Engine..." : explanation}
      </div>

      <button 
        onClick={onReset}
        className="bg-[#1A1A1A] text-white px-10 py-4 text-[11px] font-bold tracking-[0.2em] uppercase hover:bg-black transition-colors"
      >
        Reset Classification Input
      </button>
    </motion.div>
  );
};

export default function App() {
  const [data, setData] = useState<ProcessedStudent[]>([]);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [model, setModel] = useState<StudentRiskModel | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'predict'>('dashboard');

  const [formData, setFormData] = useState({
    studytime: 2,
    absences: 4,
    failures: 0,
    G1: 12,
    G2: 11,
    schoolsup: 'no',
    famsup: 'no',
    paid: 'no',
    internet: 'yes',
    Dalc: 1,
    Walc: 1,
    romantic: 'no'
  });
  const [prediction, setPrediction] = useState<{ prediction: number, probability: number } | null>(null);

  useEffect(() => {
    async function init() {
      try {
        const rawData = await fetchAndProcessData();
        setData(rawData);
        const encoded = encodeData(rawData);
        const { train, test } = splitData(encoded);
        const predictor = new StudentRiskModel();
        predictor.train(train);
        const results = predictor.evaluate(test);
        setModel(predictor);
        setMetrics(results);
      } catch (err) {
        console.error("Init fail:", err);
      } finally {
        setLoading(false);
      }
    }
    init();
  }, []);

  const handlePredict = (e: React.FormEvent) => {
    e.preventDefault();
    if (!model) return;
    const avgScore = (formData.G1 + formData.G2) / 2;
    const features = [formData.studytime, formData.absences, formData.failures, formData.G1, formData.G2, avgScore, formData.schoolsup === 'yes' ? 1 : 0, formData.famsup === 'yes' ? 1 : 0, formData.paid === 'yes' ? 1 : 0, formData.internet === 'yes' ? 1 : 0, formData.Dalc, formData.Walc, formData.romantic === 'yes' ? 1 : 0, formData.studytime * formData.absences];
    setPrediction(model.predict(features));
  };

  const chartData = useMemo(() => {
    if (data.length === 0) return { absencesVsRisk: [], failures: [], studyTimeVsRisk: [], alcoholVsRisk: [], supportFactors: [] };

    // 1. Absences vs Risk
    const absenceGroups = [{ range: '0-5', count: 0, risky: 0 }, { range: '6-10', count: 0, risky: 0 }, { range: '11-20', count: 0, risky: 0 }, { range: '21+', count: 0, risky: 0 }];
    data.forEach(s => {
      let group;
      if (s.absences <= 5) group = absenceGroups[0];
      else if (s.absences <= 10) group = absenceGroups[1];
      else if (s.absences <= 20) group = absenceGroups[2];
      else group = absenceGroups[3];
      group.count++;
      if (s.is_at_risk) group.risky++;
    });
    const absencesVsRisk = absenceGroups.map(g => ({ name: g.range, risk: g.count > 0 ? (g.risky / g.count) * 100 : 0 }));

    // 2. Failures
    const failures = [0, 1, 2, 3].map(f => ({ 
      name: `${f} Prev Failures`, 
      risky: data.filter(s => s.failures === f && s.is_at_risk).length 
    }));

    // 3. Study Time vs Risk
    const studyTimeGroups = [
      { name: '<2h', level: 1, count: 0, risky: 0 },
      { name: '2-5h', level: 2, count: 0, risky: 0 },
      { name: '5-10h', level: 3, count: 0, risky: 0 },
      { name: '10h+', level: 4, count: 0, risky: 0 }
    ];
    data.forEach(s => {
      const group = studyTimeGroups.find(g => g.level === s.studytime);
      if (group) {
        group.count++;
        if (s.is_at_risk) group.risky++;
      }
    });
    const studyTimeVsRisk = studyTimeGroups.map(g => ({
      name: g.name,
      Risk: g.count > 0 ? (g.risky / g.count) * 100 : 0
    }));

    // 4. Alcohol vs Risk (Workday)
    const alcoholLevels = [1, 2, 3, 4, 5].map(level => {
      const group = data.filter(s => s.Dalc === level);
      return {
        name: `Lvl ${level}`,
        Risk: group.length > 0 ? (group.filter(s => s.is_at_risk).length / group.length) * 100 : 0
      };
    });

    // 5. Support Factors
    const supportFactors = [
      { name: 'School Support', key: 'schoolsup' },
      { name: 'Parent Support', key: 'famsup' },
      { name: 'Internet Access', key: 'internet' },
      { name: 'Romantic Rel.', key: 'romantic' }
    ].map(f => {
      const yesGroup = data.filter(s => (s as any)[f.key] === 'yes');
      const noGroup = data.filter(s => (s as any)[f.key] === 'no');
      return {
        name: f.name,
        'Yes Risk': yesGroup.length > 0 ? (yesGroup.filter(s => s.is_at_risk).length / yesGroup.length) * 100 : 0,
        'No Risk': noGroup.length > 0 ? (noGroup.filter(s => s.is_at_risk).length / noGroup.length) * 100 : 0
      };
    });

    return { absencesVsRisk, failures, studyTimeVsRisk, alcoholVsRisk: alcoholLevels, supportFactors };
  }, [data]);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#F9F7F2] flex items-center justify-center p-4">
        <div className="text-center font-serif italic text-2xl animate-pulse">
          Calibrating Predictiva Interface...
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#F9F7F2] text-[#1A1A1A] font-sans selection:bg-black selection:text-white">
      {/* Editorial Nav */}
      <nav className="flex justify-between items-center px-6 md:px-10 py-8 border-b border-black/10">
        <div className="flex flex-col">
          <span className="text-[12px] font-bold tracking-[0.3em] uppercase">Predictiva.ED</span>
          <span className="text-[10px] opacity-40 uppercase tracking-widest">Early Warning Interface / v1.0.4</span>
        </div>
        <div className="flex gap-4 md:gap-8 text-[11px] font-bold tracking-[0.2em] uppercase">
          <button 
            onClick={() => setActiveTab('dashboard')}
            className={cn("transition-all", activeTab === 'dashboard' ? "underline underline-offset-8" : "opacity-40 hover:opacity-100")}
          >
            Dashboard
          </button>
          <button 
            onClick={() => setActiveTab('predict')}
            className={cn("transition-all", activeTab === 'predict' ? "underline underline-offset-8" : "opacity-40 hover:opacity-100")}
          >
            Predictor
          </button>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto border-x border-black/10 min-h-[calc(100vh-160px)]">
        <AnimatePresence mode="wait">
          {activeTab === 'dashboard' ? (
            <motion.div 
              key="dashboard"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="grid grid-cols-1 md:grid-cols-12"
            >
              {/* Left Insight Col */}
              <div className="md:col-span-4 border-r border-black/10 p-10">
                <h2 className="font-serif text-4xl italic mb-12">General Overview</h2>
                <div className="grid grid-cols-1 gap-0">
                  <StatCard title="Sample Size" value={data.length} icon={Users} color="" />
                  <StatCard title="Recall Rate" value={`${(metrics?.recall! * 100).toFixed(1)}%`} icon={TrendingUp} color="" />
                  <StatCard title="Precision" value={`${(metrics?.precision! * 100).toFixed(1)}%`} icon={CheckCircle2} color="" />
                </div>
                <div className="mt-12 p-6 bg-white/40 border border-black/10">
                  <h4 className="text-[10px] font-bold uppercase tracking-widest mb-4">Intervention Priority</h4>
                   <ul className="text-[13px] space-y-4 font-serif italic">
                    <li className="flex gap-4 items-start"><span className="not-italic font-bold opacity-30 text-[10px]">01.</span> Counseling requirement for high-absence profiles</li>
                    <li className="flex gap-4 items-start"><span className="not-italic font-bold opacity-30 text-[10px]">02.</span> Mathematics core support prioritization</li>
                    <li className="flex gap-4 items-start"><span className="not-italic font-bold opacity-30 text-[10px]">03.</span> Alcohol consumption risk correlation alert</li>
                  </ul>
                </div>
              </div>

              {/* Main Dashboard Grid */}
              <div className="md:col-span-8 p-10 space-y-12 bg-white/10 overflow-y-auto max-h-[calc(100vh-160px)] scrollbar-hide">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                  <div>
                    <h3 className="text-[10px] font-bold tracking-widest uppercase mb-6 opacity-60">Absence vs Risk Rate</h3>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData.absencesVsRisk}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} strokeOpacity={0.1} />
                          <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fontSize: 10}} />
                          <YAxis axisLine={false} tickLine={false} tick={{fontSize: 10}} label={{ value: 'Risk %', angle: -90, position: 'insideLeft', fontSize: 10, offset: -5 }} />
                          <Tooltip contentStyle={{ fontSize: '10px', borderRadius: '0px', border: '1px solid #eee' }} />
                          <Bar dataKey="risk" fill="#1A1A1A" radius={[2, 2, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div>
                    <h3 className="text-[10px] font-bold tracking-widest uppercase mb-6 opacity-60">Engine Diagnostics</h3>
                    <div className="space-y-6">
                      {[
                        { label: "Recall (Fail Class)", val: metrics?.recall },
                        { label: "Precision", val: metrics?.precision },
                        { label: "F1 Score", val: metrics?.f1 }
                      ].map(metric => (
                        <div key={metric.label}>
                          <div className="flex justify-between text-[11px] mb-2 font-bold uppercase tracking-widest">
                            <span>{metric.label}</span>
                            <span className="font-mono">{(metric.val! * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full h-[1px] bg-black/10">
                            <motion.div 
                              initial={{ width: 0 }}
                              animate={{ width: `${(metric.val! * 100)}%` }}
                              className="h-full bg-black"
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                  <div className="p-8 border border-black/10 bg-white/40">
                    <h3 className="text-[10px] font-bold tracking-widest uppercase mb-6 opacity-60">Study Intensity Impact</h3>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData.studyTimeVsRisk} layout="vertical">
                          <XAxis type="number" hide />
                          <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{fontSize: 10}} width={60} />
                          <Tooltip cursor={{fill: '#f1f1f1'}} contentStyle={{ fontSize: '10px', borderRadius: '0px' }} />
                          <Bar dataKey="Risk" fill="#de6b51" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <p className="text-[9px] mt-4 opacity-40 uppercase tracking-widest">Diminished risk observed at 5h+ thresholds</p>
                  </div>

                  <div className="p-8 border border-black/10 bg-white/40">
                    <h3 className="text-[10px] font-bold tracking-widest uppercase mb-6 opacity-60">Alcohol Use Correlation</h3>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData.alcoholVsRisk}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} strokeOpacity={0.1} />
                          <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fontSize: 10}} />
                          <YAxis hide />
                          <Tooltip contentStyle={{ fontSize: '10px', borderRadius: '0px' }} />
                          <Bar dataKey="Risk" fill="#1A1A1A" radius={[0, 0, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <p className="text-[9px] mt-4 opacity-40 uppercase tracking-widest">Lvl 1-5 Frequency (Workday Consump.)</p>
                  </div>
                </div>

                <div className="p-10 border border-black/10 bg-white/40">
                  <div className="flex justify-between items-center mb-8">
                    <h3 className="font-serif text-2xl italic">Systemic Support & Relationships</h3>
                    <span className="text-[10px] font-bold tracking-widest opacity-30 uppercase">Comparative Risk (%)</span>
                  </div>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData.supportFactors} barGap={8}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} strokeOpacity={0.1} />
                        <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fontSize: 11, fontStyle: 'italic'}} />
                        <YAxis axisLine={false} tickLine={false} tick={{fontSize: 10}} />
                        <Tooltip contentStyle={{ fontSize: '10px', borderRadius: '0px', backgroundColor: '#fff' }} />
                        <Bar dataKey="Yes Risk" fill="#1A1A1A" radius={[2, 2, 0, 0]} />
                        <Bar dataKey="No Risk" fill="#e2e8f0" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="flex gap-6 mt-6">
                    <div className="flex items-center gap-2"><div className="w-3 h-3 bg-[#1A1A1A] rounded-full" /><span className="text-[9px] font-bold uppercase tracking-widest opacity-60">Has Variable (Yes)</span></div>
                    <div className="flex items-center gap-2"><div className="w-3 h-3 bg-[#e2e8f0] rounded-full" /><span className="text-[9px] font-bold uppercase tracking-widest opacity-60">Lacks Variable (No)</span></div>
                  </div>
                </div>

                <div className="p-10 border border-black/10 bg-white/40">
                  <div className="flex justify-between items-center mb-8">
                    <h3 className="font-serif text-2xl italic">Failure Distribution (Past Scores)</h3>
                    <History className="w-5 h-5 opacity-20" />
                  </div>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData.failures}>
                        <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fontSize: 11, fontStyle: 'italic'}} />
                        <YAxis hide />
                        <Tooltip cursor={{fill: '#f1f1f1'}} contentStyle={{ fontSize: '10px', borderRadius: '0px' }} />
                        <Bar dataKey="risky" fill="#1A1A1A" radius={[0, 0, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div 
              key="predict"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="grid grid-cols-1 md:grid-cols-12"
            >
              <div className="md:col-span-4 p-10 border-r border-black/10">
                <h2 className="font-serif text-3xl italic mb-10">Feature Contextualization</h2>
                <p className="text-[13px] opacity-60 leading-relaxed mb-10 font-serif">
                  Input variables must be sourced from validated educational records. The system utilizes a standard scaler interaction model.
                </p>
                <div className="text-[9px] uppercase tracking-widest opacity-40 space-y-2">
                  <p>Model: Logistic Regression</p>
                  <p>Weighting: Balanced Class Distribution</p>
                  <p>Validation: 20% Test Split</p>
                </div>
              </div>

              <div className="md:col-span-8 p-10 bg-white/20">
                <AnimatePresence mode="wait">
                  {prediction ? (
                    <PredictionResult 
                      result={prediction} 
                      features={formData} 
                      onReset={() => setPrediction(null)} 
                    />
                  ) : (
                    <form onSubmit={handlePredict} className="max-w-2xl mx-auto space-y-12">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-10">
                        <div className="border-b border-black/10 pb-4">
                          <label className="block text-[10px] uppercase tracking-[0.2em] mb-4 opacity-50 font-bold">Weekly Study Effort</label>
                          <select 
                            value={formData.studytime}
                            onChange={e => setFormData({...formData, studytime: parseInt(e.target.value)})}
                            className="w-full bg-transparent font-serif italic text-xl outline-none appearance-none"
                          >
                            <option value={1}>Restricted ({"< 2h"})</option>
                            <option value={2}>Standard (2-5h)</option>
                            <option value={3}>Extended (5-10h)</option>
                            <option value={4}>Academic High (10h+)</option>
                          </select>
                        </div>
                        <div className="border-b border-black/10 pb-4">
                          <label className="block text-[10px] uppercase tracking-[0.2em] mb-4 opacity-50 font-bold">Academic Absences</label>
                          <input 
                            type="number" min="0" value={formData.absences}
                            onChange={e => setFormData({...formData, absences: parseInt(e.target.value)})}
                            className="w-full bg-transparent font-serif italic text-xl outline-none"
                          />
                        </div>
                        <div className="border-b border-black/10 pb-4">
                          <label className="block text-[10px] uppercase tracking-[0.2em] mb-4 opacity-50 font-bold">Grade 1 Performance</label>
                          <div className="flex items-center gap-2">
                            <input 
                              type="number" min="0" max="20" value={formData.G1}
                              onChange={e => setFormData({...formData, G1: parseInt(e.target.value)})}
                              className="w-full bg-transparent font-serif italic text-xl outline-none"
                            />
                            <span className="text-[10px] opacity-30 tracking-widest">/ 20.0</span>
                          </div>
                        </div>
                        <div className="border-b border-black/10 pb-4">
                          <label className="block text-[10px] uppercase tracking-[0.2em] mb-4 opacity-50 font-bold">Grade 2 Performance</label>
                           <div className="flex items-center gap-2">
                            <input 
                              type="number" min="0" max="20" value={formData.G2}
                              onChange={e => setFormData({...formData, G2: parseInt(e.target.value)})}
                              className="w-full bg-transparent font-serif italic text-xl outline-none"
                            />
                            <span className="text-[10px] opacity-30 tracking-widest">/ 20.0</span>
                          </div>
                        </div>
                        <div className="border-b border-black/10 pb-4">
                          <label className="block text-[10px] uppercase tracking-[0.2em] mb-4 opacity-50 font-bold">Past Course Failures</label>
                           <input 
                              type="number" min="0" max="3" value={formData.failures}
                              onChange={e => setFormData({...formData, failures: parseInt(e.target.value)})}
                              className="w-full bg-transparent font-serif italic text-xl outline-none"
                            />
                        </div>
                        <div className="border-b border-black/10 pb-4">
                          <label className="block text-[10px] uppercase tracking-[0.2em] mb-4 opacity-50 font-bold">Support Intensity</label>
                          <div className="flex gap-8 pt-2">
                            {['yes', 'no'].map(opt => (
                              <label key={opt} className="flex items-center gap-3 cursor-pointer group">
                                <input 
                                  type="radio" 
                                  checked={formData.schoolsup === opt}
                                  onChange={() => setFormData({...formData, schoolsup: opt})}
                                  className="accent-black"
                                />
                                <span className={cn("text-[11px] uppercase tracking-widest font-bold transition-opacity", formData.schoolsup === opt ? "opacity-100" : "opacity-30")}>{opt}</span>
                              </label>
                            ))}
                          </div>
                        </div>
                      </div>

                      <div className="pt-12">
                        <button 
                          type="submit"
                          className="w-full bg-[#1A1A1A] text-white py-6 text-[12px] font-bold tracking-[0.4em] uppercase hover:bg-black transition-all shadow-xl shadow-black/10"
                        >
                          Execute Predictive Algorithm
                        </button>
                      </div>
                    </form>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="px-10 py-10 border-t border-black/10 flex flex-col md:flex-row justify-between items-center gap-4 text-[10px] uppercase tracking-[0.2em]">
        <div className="opacity-60">UCI Student Performance Dataset Analyser</div>
        <div className="italic opacity-30 text-[9px] font-serif">Critical Threshold: G3 {"<"} 10.0 (BINARY)</div>
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
          <span>System Online: Node-AI v.44</span>
        </div>
      </footer>
    </div>
  );
}
