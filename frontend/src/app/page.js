"use client";

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import { Search, Send, Orbit, MessageSquare, X, Database, History, Link as LinkIcon, Settings, UploadCloud, FileText, CheckCircle2, AlertCircle, Paperclip } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import dynamic from 'next/dynamic';

// Dynamically import react-force-graph-2d to avoid SSR window errors
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  
  const [atoms, setAtoms] = useState([]);
  const [ledger, setLedger] = useState([]);
  
  const [highlightedAtom, setHighlightedAtom] = useState(null);
  const [highlightedMsgIdx, setHighlightedMsgIdx] = useState(null);
  const [activeModal, setActiveModal] = useState(null); 

  const [inferenceDepth, setInferenceDepth] = useState(5);
  const [persistence, setPersistence] = useState(true);

  // Document upload state
  const [uploadingDoc, setUploadingDoc] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const fileInputRef = useRef(null);

  const fetchGraphData = async () => {
    try {
      const graphRes = await fetch("http://localhost:8000/api/graph");
      const graphData = await graphRes.json();
      setAtoms(graphData.atoms || []);
      
      const ledgerRes = await fetch("http://localhost:8000/api/ledger?limit=20");
      const ledgerData = await ledgerRes.json();
      setLedger(ledgerData.log || []);
    } catch (e) {
      console.error("Failed to sync graph:", e);
    }
  };

  useEffect(() => {
    fetchGraphData();
  }, [messages]);

  const handleDocumentUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadingDoc(true);
    setUploadStatus({ type: "info", message: `Reading and extracting facts from ${file.name}...` });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/api/upload-doc", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Document ingestion failed.");
      }

      const data = await res.json();
      setUploadStatus({
        type: "success",
        message: `Parsed "${data.filename}" (${data.characters_extracted} chars). Ingested into AtomSpace (total ${data.total_atoms} atoms).`
      });

      // Inject a record into the chat conversation
      setMessages((prev) => [
        ...prev,
        { role: "user", content: `Uploaded document: ${data.filename}` },
        { role: "assistant", content: `I have read and digested "${data.filename}". All facts and relations have been verified, grounded into MeTTa atoms, and placed in your Knowledge Graph.\n\nSummary:\n${data.summary}` }
      ]);
      setIsChatOpen(true);
      fetchGraphData();
    } catch (err) {
      console.error("Document upload error:", err);
      setUploadStatus({
        type: "error",
        message: err.message || "Failed to process document"
      });
    } finally {
      setUploadingDoc(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleSend = async (overrideInput = null) => {
    const text = overrideInput || input;
    if (!text.trim()) return;

    const userMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setIsChatOpen(true);
    setHighlightedAtom(null);
    setHighlightedMsgIdx(null);

    try {
      const res = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage.content, history: messages }),
      });
      const data = await res.json();
      const assistantMessage = { role: "assistant", content: data.reply };
      setMessages((prev) => [...prev, assistantMessage]);
      fetchGraphData();
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Connection to knowledge engine failed. Is port 8000 running?" }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleAtomClick = (atomString) => {
    setHighlightedAtom(atomString);
    setIsChatOpen(true);

    const clean = atomString.replace(/[()]/g, "").trim();
    const parts = clean.split(/\s+/).filter(p => p.length > 2);
    
    // First, check if this atom matches any origin user_input from the ledger
    const matchingLedgerEntry = ledger.find(entry => {
      const diff = typeof entry === "object" ? (entry.diff || "") : String(entry);
      return diff.includes(clean) || clean.includes(diff.replace(/[()]/g, "").trim());
    });

    let targetText = matchingLedgerEntry && typeof matchingLedgerEntry === "object" ? matchingLedgerEntry.user_input : null;

    let matchedIdx = -1;
    let highestScore = 0;

    for (let i = 0; i < messages.length; i++) {
      const msgText = messages[i].content.toLowerCase();
      
      // If we found an exact origin utterance from the ledger, prioritize that user message
      if (targetText && messages[i].role === "user" && (msgText.includes(targetText.toLowerCase().trim()) || targetText.toLowerCase().includes(msgText.trim()))) {
        matchedIdx = i;
        break;
      }

      // Count term overlap
      let score = 0;
      for (const part of parts) {
        if (msgText.includes(part.toLowerCase())) {
          score += 1;
        }
      }

      // We want the most specific match, giving priority to earlier messages if score is high
      if (score > highestScore) {
        highestScore = score;
        matchedIdx = i;
      }
    }

    if (matchedIdx !== -1 && (highestScore > 0 || targetText)) {
      setHighlightedMsgIdx(matchedIdx);
      setTimeout(() => {
        const el = document.getElementById(`chat-msg-${matchedIdx}`);
        if (el) {
          el.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, 100);
    } else {
      setHighlightedMsgIdx(null);
    }
  };

  // Convert MeTTa atoms into Node-Link graph for ForceGraph2D
  const graphData = useMemo(() => {
    const nodesMap = new Map();
    const linksArray = [];

    atoms.forEach((atom) => {
      const clean = atom.replace(/[()]/g, "").trim();
      const parts = clean.split(/\s+/);

      if (parts.length >= 3) {
        const pred = parts[0];
        const arg1 = parts[1];
        const arg2 = parts.slice(2).join(" ");

        if (!nodesMap.has(arg1)) nodesMap.set(arg1, { id: arg1, group: 1 });
        if (!nodesMap.has(arg2)) nodesMap.set(arg2, { id: arg2, group: 2 });
        linksArray.push({ source: arg1, target: arg2, name: pred });
      } else if (parts.length === 2) {
        const pred = parts[0];
        const arg1 = parts[1];
        if (!nodesMap.has(arg1)) nodesMap.set(arg1, { id: arg1, group: 1 });
        if (!nodesMap.has(pred)) nodesMap.set(pred, { id: pred, group: 3 });
        linksArray.push({ source: arg1, target: pred, name: "is-a" });
      }
    });

    return { nodes: Array.from(nodesMap.values()), links: linksArray };
  }, [atoms]);

  return (
    <main className="min-h-screen bg-gradient-to-br from-[#f3e8ff] via-[#e0e7ff] to-[#fce7f3] text-[#1f2937] font-sans relative overflow-hidden flex flex-col">
      
      {/* -----------------------------------------------------------------
          Top Navigation Bar
          ----------------------------------------------------------------- */}
      <nav className="flex items-center justify-between px-8 py-6 max-w-[1400px] mx-auto w-full z-10 relative">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-purple-600 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-purple-500/30 text-white cursor-pointer" onClick={() => {setMessages([]); setIsChatOpen(false);}}>
            <Orbit size={20} />
          </div>
          <span className="text-xl font-bold tracking-tight text-gray-900">Super memory</span>
        </div>

        <div className="hidden md:flex items-center gap-6 text-sm font-medium text-gray-500">
          <button onClick={() => setActiveModal("settings")} className="transition-colors hover:text-gray-900 flex items-center gap-2 px-3 py-1.5 rounded-full hover:bg-white/50">
            <Settings size={16}/> Graph Settings
          </button>
          
          {/* Text Document Upload Trigger */}
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleDocumentUpload} 
            accept=".txt,.md,.csv,.log,.json,text/plain,text/markdown,text/csv" 
            className="hidden" 
          />
          <button 
            onClick={() => fileInputRef.current?.click()} 
            disabled={uploadingDoc}
            className="flex items-center gap-2 transition-all hover:bg-white/70 px-3.5 py-1.5 rounded-full bg-white/40 text-purple-700 font-semibold border border-purple-200/50 shadow-sm"
          >
            <UploadCloud size={16} className={uploadingDoc ? "animate-pulse" : ""}/>
            {uploadingDoc ? "Reading Text..." : "Upload Text Doc"}
          </button>

          <button className="flex items-center gap-1 transition-colors text-purple-600 font-semibold px-3 py-1.5 rounded-full bg-purple-100/50">
             <span className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_8px_#22c55e]"></span> Active Engine
          </button>
        </div>

        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-full bg-gray-200 border-2 border-white shadow-sm flex items-center justify-center text-gray-500">
             <span className="font-bold text-sm">SM</span>
          </div>
        </div>
      </nav>

      {/* -----------------------------------------------------------------
          Modals Overlay
          ----------------------------------------------------------------- */}
      <AnimatePresence>
        {activeModal === "settings" && (
          <div className="fixed inset-0 z-[100] bg-[#09090b]">
            
            {/* The Live Interactive Graph Background */}
            <div className="absolute inset-0 z-0">
               <ForceGraph2D
                  graphData={graphData}
                  nodeLabel="id"
                  nodeColor={(node) => node.group === 1 ? '#a855f7' : node.group === 2 ? '#ec4899' : '#3b82f6'}
                  nodeRelSize={8}
                  linkColor={() => 'rgba(255,255,255,0.2)'}
                  linkWidth={2}
                  backgroundColor="#09090b"
               />
            </div>

            {/* Clear, Prominent Return Button (Top Right) */}
            <button 
              onClick={() => setActiveModal(null)} 
              className="absolute top-8 right-8 z-50 flex items-center gap-2.5 px-5 py-2.5 rounded-full bg-white/10 hover:bg-white text-gray-200 hover:text-gray-900 border border-white/20 backdrop-blur-xl shadow-2xl font-semibold text-sm transition-all duration-200 hover:scale-105 active:scale-95 group"
            >
              <span>Back to Dashboard</span>
              <X size={18} className="transition-transform group-hover:rotate-90 text-gray-400 group-hover:text-gray-900" />
            </button>

            {/* Left Glass Floating Controls Panel */}
            <motion.div 
              initial={{ x: -100, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -100, opacity: 0 }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="absolute top-10 left-10 w-96 bg-white/10 backdrop-blur-2xl border border-white/10 rounded-3xl shadow-2xl p-8 z-10 text-gray-100"
            >
              <button onClick={() => setActiveModal(null)} className="absolute top-6 right-6 text-gray-400 hover:text-white p-1 rounded-full hover:bg-white/10 transition-colors">
                <X size={20}/>
              </button>
              
              <div className="flex items-center gap-3 mb-8">
                <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center text-purple-400"><Settings size={20}/></div>
                <h2 className="text-xl font-bold">Graph Settings</h2>
              </div>
              
              <div className="space-y-8">
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm font-semibold text-gray-300">Inference Depth (Hops)</span>
                    <span className="text-sm font-bold text-purple-400">{inferenceDepth}</span>
                  </div>
                  <input type="range" min="1" max="10" value={inferenceDepth} onChange={(e)=>setInferenceDepth(e.target.value)} className="w-full accent-purple-500" />
                  <p className="text-xs text-gray-400 mt-2">Controls how deep the backward-chaining resolver digs for logical proofs in this visual network.</p>
                </div>

                <div className="flex items-center justify-between p-4 bg-white/5 rounded-2xl border border-white/5">
                  <div>
                    <span className="text-sm font-semibold text-gray-200 block">SQLite Disk Persistence</span>
                    <span className="text-xs text-gray-400">Persist evidence ledger to disk.</span>
                  </div>
                  <button onClick={()=>setPersistence(!persistence)} className={`w-12 h-6 rounded-full p-1 transition-colors ${persistence ? 'bg-purple-500' : 'bg-white/20'}`}>
                    <div className={`w-4 h-4 bg-white rounded-full shadow-sm transition-transform ${persistence ? 'translate-x-6' : 'translate-x-0'}`}></div>
                  </button>
                </div>

                <div className="mt-8 pt-6 border-t border-white/10">
                   <h3 className="text-xs font-bold uppercase text-gray-500 tracking-wider mb-3">Graph Stats</h3>
                   <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Total Nodes</span>
                      <span className="text-white font-mono">{graphData.nodes.length}</span>
                   </div>
                   <div className="flex justify-between text-sm mt-1">
                      <span className="text-gray-400">Connections</span>
                      <span className="text-white font-mono">{graphData.links.length}</span>
                   </div>
                </div>
              </div>
            </motion.div>
            
            <div className="absolute bottom-10 right-10 text-white/30 text-sm font-mono tracking-widest pointer-events-none">
              INTERACTIVE 2D KNOWLEDGE GRAPH
            </div>
          </div>
        )}
      </AnimatePresence>

      {/* -----------------------------------------------------------------
          Main Canvas Center Area
          ----------------------------------------------------------------- */}
      <div className="flex-1 w-full max-w-[1400px] mx-auto relative px-8 pb-8 z-10">
        
        <AnimatePresence mode="wait">
          {messages.length === 0 ? (
            /* INITIAL STATE: Hero Input */
            <motion.div 
              key="hero"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, filter: "blur(10px)" }}
              transition={{ duration: 0.4 }}
              className="flex flex-col items-center justify-center h-[70vh] w-full max-w-3xl mx-auto"
            >
              <h1 className="text-4xl font-semibold tracking-tight text-gray-900 mb-2">Hi, I'm Super memory.</h1>
              <p className="text-gray-500 mb-6 text-lg">Send a message or upload a text document to extract, ground, and retain knowledge.</p>

              {/* Status Banner */}
              {uploadStatus && (
                <div className={`mb-6 px-4 py-2.5 rounded-2xl flex items-center gap-2 text-xs font-medium border ${
                  uploadStatus.type === "success" 
                    ? "bg-green-50 text-green-800 border-green-200" 
                    : uploadStatus.type === "error" 
                    ? "bg-red-50 text-red-800 border-red-200" 
                    : "bg-purple-50 text-purple-800 border-purple-200 animate-pulse"
                }`}>
                  {uploadStatus.type === "success" ? <CheckCircle2 size={16}/> : uploadStatus.type === "error" ? <AlertCircle size={16}/> : <FileText size={16}/>}
                  {uploadStatus.message}
                </div>
              )}

              <div className="w-full bg-white/40 backdrop-blur-xl border border-white/60 rounded-3xl p-4 shadow-[0_8px_32px_rgba(0,0,0,0.05)] transition-all focus-within:bg-white/60">
                <div className="flex items-center mb-4">
                  <input 
                    type="text" 
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSend()}
                    placeholder="Teach me a fact, ask a question, or upload a text file..." 
                    className="w-full bg-transparent border-none focus:outline-none text-xl placeholder-gray-400 px-2 py-4 text-gray-900"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <button 
                      onClick={() => fileInputRef.current?.click()} 
                      disabled={uploadingDoc}
                      className="px-3.5 py-1.5 rounded-full bg-white/80 hover:bg-white text-purple-700 text-xs font-semibold flex items-center gap-1.5 transition-all shadow-sm border border-purple-200/50"
                    >
                      <UploadCloud size={14} className={uploadingDoc ? "animate-pulse" : ""}/>
                      {uploadingDoc ? "Reading..." : "Upload Text Doc"}
                    </button>
                    <button onClick={() => handleSend("Bob used to work at Google, but he works at OpenAI now. Where does Bob work?")} className="px-3 py-1.5 rounded-full bg-white/50 hover:bg-white text-gray-600 text-xs font-medium transition-colors shadow-sm">Belief Revision Test</button>
                    <button onClick={() => handleSend("Alice is married to Charlie. Who is Charlie married to?")} className="px-3 py-1.5 rounded-full bg-white/50 hover:bg-white text-gray-600 text-xs font-medium transition-colors shadow-sm">Symmetric Relation Test</button>
                  </div>
                  {input.trim() && (
                     <button onClick={() => handleSend()} className="w-10 h-10 rounded-full bg-purple-600 hover:bg-purple-700 flex items-center justify-center text-white shadow-lg shadow-purple-500/30">
                       <Send size={18} />
                     </button>
                  )}
                </div>
              </div>
            </motion.div>

          ) : (
            /* ACTIVE STATE: Live Dashboard */
            <motion.div 
              key="dashboard"
              initial={{ opacity: 0, filter: "blur(10px)" }}
              animate={{ opacity: 1, filter: "blur(0px)" }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="w-full h-full grid grid-cols-2 gap-8 pr-[420px]" 
            >
              
              {/* Left Column: Live AtomSpace */}
              <div className="bg-white/40 backdrop-blur-xl border border-white/60 rounded-3xl p-6 shadow-[0_8px_32px_rgba(0,0,0,0.05)] flex flex-col h-[75vh]">
                <div className="flex items-center justify-between pb-4 border-b border-gray-200/40">
                  <div className="flex items-center gap-2">
                    <Database size={18} className="text-purple-600" />
                    <h2 className="font-semibold text-gray-800 text-base">Live AtomSpace</h2>
                  </div>
                  <span className="text-xs bg-purple-100 text-purple-700 font-semibold px-2.5 py-0.5 rounded-full">
                    {atoms.length} Atoms Grounded
                  </span>
                </div>

                <div className="flex-1 overflow-y-auto mt-4 space-y-2 pr-2 custom-scrollbar">
                  {atoms.length === 0 ? (
                    <div className="text-gray-400 text-sm flex items-center justify-center h-full">
                      No atoms loaded in memory.
                    </div>
                  ) : (
                    atoms.map((atom, idx) => (
                      <div 
                        key={idx}
                        onClick={() => handleAtomClick(atom)}
                        className={`text-xs font-mono p-3 rounded-xl border transition-all cursor-pointer select-none ${
                          highlightedAtom === atom
                            ? "bg-purple-600 text-white border-purple-600 shadow-md shadow-purple-500/20 scale-[1.01]"
                            : "bg-white/60 hover:bg-white text-gray-700 border-white/80 hover:border-purple-200 hover:shadow-sm"
                        }`}
                      >
                        {atom}
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Right Column: Ledger Audit Trail */}
              <div className="bg-white/40 backdrop-blur-xl border border-white/60 rounded-3xl p-6 shadow-[0_8px_32px_rgba(0,0,0,0.05)] flex flex-col h-[75vh]">
                <div className="flex items-center justify-between pb-4 border-b border-gray-200/40">
                  <div className="flex items-center gap-2">
                    <History size={18} className="text-fuchsia-600" />
                    <h2 className="font-semibold text-gray-800 text-base">Evidence Ledger</h2>
                  </div>
                  <span className="text-xs bg-fuchsia-100 text-fuchsia-700 font-semibold px-2.5 py-0.5 rounded-full">
                    SQLite Immutable Log
                  </span>
                </div>

                <div className="flex-1 overflow-y-auto mt-4 space-y-3 pr-2 custom-scrollbar">
                  {ledger.length === 0 ? (
                     <div className="text-gray-400 text-sm flex items-center justify-center h-full">
                       No audit entries recorded yet.
                     </div>
                  ) : (
                     ledger.map((entry, idx) => {
                        const isObj = typeof entry === "object" && entry !== null;
                        const op = isObj ? (entry.operation || "assert") : "assert";
                        const diffText = isObj ? (entry.diff || JSON.stringify(entry)) : String(entry);
                        const isRetraction = op === "remove" || op === "retract" || diffText.includes("RETRACT") || diffText.includes("REMOVED");

                        return (
                          <div 
                            key={idx}
                            className={`text-xs font-mono p-3 rounded-xl border bg-white/60 transition-all ${
                              isRetraction ? "border-amber-200/60 bg-amber-50/40 text-amber-900" : "border-white/80 text-gray-700"
                            }`}
                          >
                             <div className="flex items-center justify-between mb-1 opacity-60 text-[10px]">
                               <span>Entry #{ledger.length - idx}</span>
                               <span className={isRetraction ? "text-amber-600 font-bold" : "text-purple-600 font-bold"}>
                                 {isRetraction ? "RETRACT" : "ASSERT"}
                               </span>
                             </div>
                             <div className="font-semibold">{diffText}</div>
                             {isObj && entry.user_input && (
                               <div className="text-[10px] text-gray-400 mt-1 line-clamp-1 italic truncate">
                                 &ldquo;{entry.user_input}&rdquo;
                               </div>
                             )}
                          </div>
                        );
                     })
                  )}
                </div>
              </div>

            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* -----------------------------------------------------------------
          Floating Chat Widget (Bottom Right)
          ----------------------------------------------------------------- */}
      <AnimatePresence>
        {isChatOpen && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.9, y: 50, x: 50 }}
            animate={{ opacity: 1, scale: 1, y: 0, x: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 50, x: 50 }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className={`fixed bottom-6 right-6 w-[400px] h-[75vh] min-h-[500px] bg-white/80 backdrop-blur-2xl border border-white/60 shadow-[0_30px_60px_rgba(0,0,0,0.15)] rounded-3xl flex flex-col overflow-hidden z-50 transition-all ${highlightedAtom ? "ring-2 ring-purple-500/50 shadow-[0_0_40px_rgba(168,85,247,0.25)]" : ""}`}
          >
            <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between bg-white/60">
               <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-purple-600 to-fuchsia-500 flex items-center justify-center text-white shadow-md">
                    <Orbit size={16} />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-gray-900">Super memory</h3>
                    <p className="text-[10px] text-green-500 font-medium flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-green-500"></span> Engine Online</p>
                  </div>
               </div>
               <button onClick={() => setIsChatOpen(false)} className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center text-gray-500 hover:bg-black hover:text-white transition-colors">
                 <X size={16} />
               </button>
            </div>

            <div className="flex-1 overflow-y-auto p-5 space-y-6 flex flex-col custom-scrollbar relative">
               
               <AnimatePresence>
                 {highlightedAtom && (
                    <motion.div 
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="sticky top-0 w-full bg-purple-100/90 backdrop-blur text-purple-900 text-[11px] font-bold py-2 px-3 rounded-xl border border-purple-300 flex items-center justify-center gap-2 shadow-sm z-10 mb-4 tracking-wide uppercase"
                    >
                       <LinkIcon size={12}/> Inspecting trace for Atom
                    </motion.div>
                 )}
               </AnimatePresence>

               {messages.map((msg, i) => {
                 const isHighlighted = i === highlightedMsgIdx;
                 
                 return (
                   <div key={i} id={`chat-msg-${i}`} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} items-end gap-2 transition-all duration-300 ${isHighlighted ? "scale-[1.02]" : ""}`}>
                     {msg.role === "assistant" && (
                       <div className="w-6 h-6 rounded-full bg-gradient-to-tr from-purple-600 to-fuchsia-500 flex-shrink-0 flex items-center justify-center text-white shadow-sm mb-1">
                         <Orbit size={12} />
                       </div>
                     )}
                     <div className="flex flex-col gap-1 max-w-[80%]">
                       <div className={`px-4 py-3 text-[13px] leading-relaxed shadow-sm transition-all duration-300 ${
                         msg.role === "user" 
                           ? "bg-white border border-gray-100 text-gray-800 rounded-2xl rounded-br-sm" 
                           : `border rounded-2xl rounded-bl-sm ${isHighlighted ? "bg-purple-50 text-purple-900 border-purple-400 ring-2 ring-purple-400 shadow-[inset_0_0_15px_rgba(168,85,247,0.15)]" : "bg-gray-50/90 border-gray-100 text-gray-800"}`
                       }`}>
                         {msg.content}
                       </div>
                     </div>
                   </div>
                 );
               })}
               
               {loading && (
                 <div className="flex justify-start items-end gap-2">
                    <div className="w-6 h-6 rounded-full bg-gradient-to-tr from-purple-600 to-fuchsia-500 flex-shrink-0 flex items-center justify-center text-white shadow-sm mb-1">
                       <Orbit size={12} />
                    </div>
                    <div className="px-4 py-3 text-sm bg-gray-50/90 border border-gray-100 text-gray-500 rounded-2xl rounded-bl-sm flex items-center gap-1 shadow-sm">
                      <span className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-bounce"></span>
                      <span className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-bounce" style={{animationDelay: "0.2s"}}></span>
                      <span className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-bounce" style={{animationDelay: "0.4s"}}></span>
                    </div>
                 </div>
               )}
            </div>

            <div className="p-4 bg-white/80 backdrop-blur-md border-t border-gray-100">
               <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-1.5 flex items-center focus-within:border-purple-300 focus-within:ring-2 focus-within:ring-purple-100 transition-all">
                 <button 
                   type="button"
                   onClick={() => fileInputRef.current?.click()} 
                   disabled={uploadingDoc}
                   title="Attach text document (.txt, .md, .csv, .json)"
                   className="w-8 h-8 rounded-full flex items-center justify-center text-gray-400 hover:text-purple-600 hover:bg-purple-50 transition-colors ml-0.5 flex-shrink-0"
                 >
                   <Paperclip size={16} className={uploadingDoc ? "animate-pulse text-purple-600" : ""} />
                 </button>
                 <input 
                   type="text"
                   value={input}
                   onChange={(e) => setInput(e.target.value)}
                   onKeyDown={(e) => e.key === "Enter" && handleSend()}
                   placeholder="Type message..."
                   className="flex-1 bg-transparent border-none focus:outline-none text-[13px] px-3 py-2 text-gray-800 placeholder-gray-400"
                 />
                 <button 
                   onClick={() => handleSend()}
                   className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${
                     input.trim() ? "bg-purple-600 text-white shadow-md shadow-purple-500/30 hover:scale-105" : "bg-gray-100 text-gray-400"
                   }`}
                 >
                   <Send size={14} className={input.trim() ? "ml-0.5" : ""} />
                 </button>
               </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {!isChatOpen && messages.length > 0 && (
        <button 
          onClick={() => setIsChatOpen(true)}
          className="fixed bottom-6 right-6 w-14 h-14 rounded-full bg-gradient-to-tr from-purple-600 to-fuchsia-500 text-white flex items-center justify-center shadow-xl shadow-purple-500/40 hover:scale-105 transition-transform z-50"
        >
          <MessageSquare size={24} />
        </button>
      )}

      {/* Ambient glowing orbs */}
      <div className="fixed top-[20%] left-[10%] w-[40vw] h-[40vw] bg-purple-300/30 rounded-full blur-[100px] -z-10 mix-blend-multiply pointer-events-none"></div>
      <div className="fixed bottom-[10%] right-[20%] w-[35vw] h-[35vw] bg-fuchsia-200/40 rounded-full blur-[100px] -z-10 mix-blend-multiply pointer-events-none"></div>
      
      {/* Scrollbar CSS */}
      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent; 
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(0,0,0,0.1); 
          border-radius: 10px;
        }
        .custom-scrollbar:hover::-webkit-scrollbar-thumb {
          background: rgba(0,0,0,0.2); 
        }
      `}} />
    </main>
  );
}
