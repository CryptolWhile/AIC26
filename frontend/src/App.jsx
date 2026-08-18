import './index.css'; // Changed to index.css since styles/index.css was invalid
import React, { useRef, useState, useCallback, useMemo } from 'react';
import { Box, Typography, Grid, Input, Dialog, DialogTitle, DialogContent, IconButton, styled, CardMedia } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ElectricBoltIcon from '@mui/icons-material/ElectricBolt';
import NoResultDiv from './components/NoResultDiv';
import CustomTextField from './components/CustomTextField';
import ResultIframe from './components/ResultIframe';
import CustomButton from './components/CustomButton';
import CustomAutocomplete from './components/CustomAutocomplete';
import { IMAGE_DIR, VIDEO_DIR } from "./config/appConfig";
import { datasets, videos, rerankMethods } from './constants/options';
import MultipleSelectChip from './components/SelectChip';
import axios from "axios";

// Mock Data
const sampleResponseTrake = {
  "results": [
    [
      [
        { "image_id": "L26_V061_K3404", "is_main_frame": true, "score": 0.3365, "timestamp": 135.88 },
        { "image_id": "L26_V061_K3405", "is_main_frame": false, "score": 0.327, "timestamp": 136.16 }
      ],
      [
        { "image_id": "L26_V061_K3406", "is_main_frame": true, "score": 0.3658, "timestamp": 148.64 }
      ],
    ],
    [
      [
        { "image_id": "L27_V061_K3407", "is_main_frame": true, "score": 0.3365, "timestamp": 135.88 },
        { "image_id": "L27_V061_K3408", "is_main_frame": false, "score": 0.327, "timestamp": 136.16 }
      ],
      [
        { "image_id": "L27_V061_K3409", "is_main_frame": true, "score": 0.3658, "timestamp": 148.64 }
      ],
    ],
  ]
};

const sampleResponse = {
  "results": [
    { "image_id": "L26_V061_K3404", "score": 0.34 },
    { "image_id": "L26_V061_K3406", "score": 0.32 },
    { "image_id": "L26_V061_K3405", "score": 0.34 },
    { "image_id": "L27_V061_K3407", "score": 0.32 },
    { "image_id": "L27_V061_K3408", "score": 0.34 },
    { "image_id": "L27_V061_K3409", "score": 0.32 },
  ]
};

function formatImageId(image_id) {
  return image_id.split('_').join('/');
}

async function searchUser(query, isTrake) {
  try {
    const response = await axios.post("http://localhost:8000/api/search/", query);
    let result = [];
    if (!isTrake) {
      if (response.data.results && response.data.results.length > 0) {
        result = response.data.results[0].map((frame) => formatImageId(frame.image_id));
      }
    } else {
      if (response.data.results && response.data.results.length > 0) {
        result = response.data.results[0].map(videoGroups =>
          videoGroups.map(group => {
            const formatted = group.map(frame => ({
              ...frame,
              image_id: formatImageId(frame.image_id)
            }));
            const mainFrame = formatted.find(f => f.is_main_frame) ?? formatted[0];
            return {
              allFrames: formatted,
              mainFrame: mainFrame.image_id
            };
          })
        );
      }
    }
    return result;
  } catch (err) {
    console.error("Search error:", err);
    throw err;
  }
}

const BootstrapDialog = styled(Dialog)(({ theme }) => ({
  '& .MuiDialogContent-root': { padding: theme.spacing(3), backgroundColor: '#f8fafc' },
  '& .MuiDialogActions-root': { padding: theme.spacing(1) },
  '& .MuiDialog-paper': {
    width: '85%', maxWidth: 'none', height: '85%',
    borderRadius: 'var(--radius-lg)',
    boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)'
  },
}));

const MEDIA_INFO = '/media-info/';
function getNeighborFrames(frameName, offset = 10, setNeighborFrames, setVideoDialog, setFps) {
  let split = frameName.split('/');
  let src = MEDIA_INFO + split[0] + '_' + split[1] + '.json';
  fetch(src)
    .then((res) => res.text())
    .then((text) => {
      let data = JSON.parse(text);
      
      // Chuyển đổi watch_url thành link embed để iframe có thể phát được
      let videoUrl = '';
      if (data.watch_url) {
        videoUrl = data.watch_url.replace('watch?v=', 'embed/');
      } else if (data.videoSrc) {
        videoUrl = data.videoSrc;
      }
      setVideoDialog(videoUrl);
      setFps(data.fps || 25);

      if (data.listFrames) {
        let idx = data.listFrames.indexOf(split[2]);
        if (idx === -1) {
          setNeighborFrames([]); return;
        }
        let start = Math.max(0, idx - offset);
        let end = Math.min(data.listFrames.length, idx + offset + 1);
        let res = data.listFrames.slice(start, end).map(f => split[0] + '/' + split[1] + '/' + f);
        setNeighborFrames(res);
      } else {
        // Tránh lỗi crash nếu file JSON mới không có mảng listFrames
        setNeighborFrames([]);
      }
    })
    .catch((e) => {
      console.error(e);
      setNeighborFrames([]);
    });
}

// Extracted MainFrame out of App to prevent recreation on every render
const MainFrame = React.memo(function MainFrame({ 
  frame, 
  frameIdx, 
  inGroup = false, 
  group, 
  groupIdx, 
  videoIdx,
  allOrder,
  isTrake,
  onOpenDialogTrake,
  onOpenDialog,
  onFrameClick
}) {
  const imageSrc = IMAGE_DIR + frame + '.jpg';
  
  const handleClick = useCallback(() => {
    isTrake ? onOpenDialogTrake(group, groupIdx, frame, frameIdx, videoIdx) : onOpenDialog(frame)
  }, [isTrake, onOpenDialogTrake, group, groupIdx, frame, frameIdx, videoIdx, onOpenDialog]);

  return (
    <div onClick={handleClick}>
      <ResultIframe 
        src={imageSrc} 
        frameName={frame} 
        orderNumber={allOrder.get(frame)} 
        onNumberClick={onFrameClick} 
        inGroup={inGroup} 
      />
    </div>
  );
});

function groupFramesByVideo(frames) {
  const groups = new Map();
  for (const frame of frames) {
    const parts = frame.split('/');
    const videoId = parts[0] + '/' + parts[1];
    const frameNum = parseInt(parts[2], 10);

    if (!groups.has(videoId)) {
      groups.set(videoId, []);
    }
    groups.get(videoId).push({ frame, num: frameNum });
  }

  const sortedres = [];
  for (const arr of groups.values()) {
    arr.sort((a, b) => a.num - b.num);
    sortedres.push(arr.map(item => item.frame));
  }
  return sortedres;
}

function App() {
  const [allOrder, setAllOrder] = useState(() => new Map());
  const [groupOrder, setGroupOrder] = useState(() => new Map());
  const [result, setResult] = useState([]);
  const [nResults, setNResults] = useState("");
  const [groupedRes, setGroupedRes] = useState([]);
  const [trakeRes, setTrakeRes] = useState([]);
  const [isGroup, setIsGroup] = useState(false);
  const [isTrake, setIsTrake] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [selectedVideo, setSelectedVideo] = useState('');
  const [selectedRerank, setSelectedRerank] = useState('All');
  const [videoOptions, setVideoOptions] = useState([]);
  const currentSceneRef = useRef(), ocrSearchRef = useRef(), VQARef = useRef();
  const [extraPromptRefs, setExtraPromptRefs] = useState([]);
  const [selectedModel, setSelectedModel] = useState(['hf_clip_L']);

  // Optimizations: UseCallback for stable references
  const resetClick = useCallback(() => {
    if(currentSceneRef.current) currentSceneRef.current.value = '';
    if(ocrSearchRef.current) ocrSearchRef.current.value = ''; 
    if(VQARef.current) VQARef.current.value = '';
    setSelectedDataset(''); setSelectedVideo(''); setSelectedRerank('All'); setGroupedRes([]); setTrakeRes([]); setResult([]);
    setAllOrder(new Map()); setGroupOrder(new Map());
  }, []);

  const searchClick = useCallback(async () => {
    let traketype = (extraPromptRefs.length > 0);
    setIsTrake(traketype);
    const query = {
      "prompt": currentSceneRef.current?.value || "",
      "extra_prompt": extraPromptRefs.map((ref) => ref.current?.value || "") || [],
      "ocr_search": ocrSearchRef.current?.value || "",
      "dataset": selectedDataset || "All",
      "video": selectedVideo || "All",
      "n_results": nResults || 150,
      "model": selectedModel.length ? selectedModel : ["hf_clip_L"],
      "rerank_method": selectedRerank || "All"
    };
    const res = await searchUser(query, traketype);
    if (traketype) {
      setTrakeRes(res);
      setGroupOrder(new Map());
    } else {
      setResult(res); 
      setGroupedRes(groupFramesByVideo(res));
      setAllOrder(new Map());
    }
  }, [extraPromptRefs, selectedDataset, selectedVideo, nResults, selectedModel, selectedRerank]);

  const deselectAllClick = useCallback(() => {
    setAllOrder(new Map()); setGroupOrder(new Map());
  }, []);

  const selectAllClick = useCallback(() => {
    if (isTrake) {
      setGroupOrder(() => {
        const newMap = new Map();
        trakeRes.forEach((resultGroup, idx) => {
          newMap.set(idx, idx + 1);
        });
        return newMap;
      });
    } else {
      setAllOrder(() => {
        const newMap = new Map();
        result.forEach((frame, idx) => {
          newMap.set(frame, idx + 1);
        });
        return newMap;
      });
    }
  }, [result, isTrake, trakeRes]);

  const handleAddPrompt = useCallback(() => {
    setExtraPromptRefs(prev => [...prev, React.createRef()]);
  }, []);

  const handleDeletePrompt = useCallback((idx) => {
    setExtraPromptRefs(prev => prev.filter((_, i) => i !== idx));
  }, []);

  const [neighborFrames, setNeighborFrames] = useState([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [frameDialog, setFrameDialog] = useState('');
  const [videoDialog, setVideoDialog] = useState('');
  const [openDialogTrake, setOpenDialogTrake] = useState(false);
  const [relatedFrames, setRelatedFrames] = useState([]);
  const [groupDialog, setGroupDialog] = useState(0);
  const [mainVideoIndex, setMainVideoIndex] = useState(0);
  const [mainFrameDialog, setMainFrameDialog] = useState({});
  const [fps, setFps] = useState(25);

  const handleCloseDialog = useCallback((e) => {
    e.stopPropagation(); setOpenDialog(false);
  }, []);

  const handleOpenDialogTrake = useCallback((group, groupIdx, frame, frameIdx, videoIdx) => {
    getNeighborFrames(frame, 10, setNeighborFrames, setVideoDialog, setFps);
    setOpenDialogTrake(true); setRelatedFrames(group); setGroupDialog(groupIdx); setFrameDialog(frame); setMainFrameDialog(frame); setMainVideoIndex(videoIdx);
  }, []);

  const handleCloseDialogTrake = useCallback((e) => {
    e.stopPropagation(); setOpenDialogTrake(false);
  }, []);

  const handleOpenDialog = useCallback((frameName) => {
    getNeighborFrames(frameName, 10, setNeighborFrames, setVideoDialog, setFps);
    setOpenDialog(true); setFrameDialog(frameName);
  }, []);

  const handleRelatedClick = useCallback((e, frame) => {
    e.stopPropagation();
    setTrakeRes(prev => prev.map((videoGroups, videoIdx) => {
      if (videoIdx !== mainVideoIndex) return videoGroups;
      return videoGroups.map((group, groupIdx) => {
        if (groupIdx !== groupDialog) return group;
        return { ...group, mainFrame: frame };
      });
    })); 
    setMainFrameDialog(frame);
  }, [mainVideoIndex, groupDialog]);

  // Pure state updater
  const handleFrameClick = useCallback((e, frameName) => {
    e.stopPropagation();
    setAllOrder((prev) => {
      const newMap = new Map(prev);
      let value = newMap.get(frameName) || 0;
      if (value === 0) {
        let max = 0;
        newMap.forEach(v => { if (v > max) max = v; });
        newMap.set(frameName, max + 1);
      } else {
        newMap.delete(frameName);
        newMap.forEach((v, k) => {
          if (v > value) newMap.set(k, v - 1);
        });
      }
      return newMap;
    });
  }, []);

  const handleGroupClick = useCallback((e, groupId) => {
    e.stopPropagation();
    setGroupOrder((prev) => {
      const newMap = new Map(prev);
      let value = newMap.get(groupId) || 0;
      if (value === 0) {
        let max = 0;
        newMap.forEach(v => { if (v > max) max = v; });
        newMap.set(groupId, max + 1);
      } else {
        newMap.delete(groupId);
        newMap.forEach((v, k) => {
          if (v > value) newMap.set(k, v - 1);
        });
      }
      return newMap;
    });
  }, []);

  const handleSaveOutput = useCallback(async () => {
    if (allOrder.size === 0) {
      alert("Bạn chưa chọn khung hình nào!");
      return;
    }
    
    // Sắp xếp các ảnh theo thứ tự đã click (value của Map)
    const sortedEntries = Array.from(allOrder.entries()).sort((a, b) => a[1] - b[1]);
    
    // Chuyển đổi định dạng "K01/V01/785" thành "K01_V01,785" (Chuẩn nộp bài AIC)
    const csvLines = sortedEntries.map(([frame, order]) => {
      const parts = frame.split('/');
      return `${parts[0]}_${parts[1]},${parts[2]}`;
    });
    
    const queryName = currentSceneRef.current?.value || "query_result";
    
    try {
      await axios.post("http://localhost:8000/api/submit/", {
        [queryName]: csvLines
      });
      alert(`Đã lưu thành công ${csvLines.length} khung hình vào thư mục submission.zip của Backend!`);
    } catch (err) {
      console.error(err);
      alert("Có lỗi xảy ra khi lưu Output. Hãy kiểm tra Backend!");
    }
  }, [allOrder]);

  const widthDiv = '320px';

  return (
    <Box sx={{ display: 'flex', flexDirection: 'row', width: '100%', height: '100vh', padding: '16px', gap: '20px', boxSizing: 'border-box' }}>
      
      {/* LEFT SIDEBAR - Glassmorphism */}
      <Box className="glass-panel" sx={{ width: widthDiv, display: 'flex', flexDirection: 'column', borderRadius: 'var(--radius-lg)', overflowY: 'scroll', padding: '24px' }}>
        <Typography variant='h4' sx={{ color: 'var(--primary-color)', fontWeight: 700, marginBottom: 3, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          AIO - The Winner
        </Typography>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '100%' }}>
          <CustomTextField label="Primary Prompt" inputRef={currentSceneRef} multiline rows={3} />
          <CustomButton title="+ Add Action Sequence" size="large" onClick={handleAddPrompt} />
          
          {extraPromptRefs.map((ref, idx) => (
            <Box key={idx} sx={{ display: 'flex', gap: 1, alignItems: 'flex-start' }}>
              <CustomTextField label={`Sequence ${idx + 1}`} inputRef={ref} sx={{ flex: 1 }} multiline rows={2} />
              <IconButton onClick={() => handleDeletePrompt(idx)} sx={{ bgcolor: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', '&:hover': { bgcolor: 'rgba(239, 68, 68, 0.2)' } }}>
                <CloseIcon />
              </IconButton>
            </Box>
          ))}
          
          <CustomTextField label="OCR Search" inputRef={ocrSearchRef} />
          
          <Box sx={{ display: 'flex', gap: 2 }}>
            <CustomAutocomplete width='50%' options={datasets} label="Dataset" value={selectedDataset} onChange={(_, v) => { setSelectedDataset(v); setVideoOptions(videos[v] || []) }} />
            <CustomAutocomplete width='50%' options={videoOptions} label="Video" value={selectedVideo} onChange={(_, v) => setSelectedVideo(v)} freeSolo onInputChange={(_, newInputValue) => setSelectedVideo(newInputValue)} />
          </Box>
          
          <CustomAutocomplete width='100%' options={rerankMethods} label="Rerank Method" value={selectedRerank} onChange={(_, v) => setSelectedRerank(v)} />
          
          <Box onClick={(e) => e.stopPropagation()}>
            <MultipleSelectChip selectedModel={selectedModel} setSelectedModel={setSelectedModel} />
          </Box>

          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mt: 1 }}>
            <CustomTextField label="N max" type="number" sx={{ width: '65px', minWidth: '65px', flexShrink: 0 }} value={nResults} onChange={(e) => setNResults(e.target.value)} />
            <CustomButton title="Reset" size="medium" onClick={resetClick} sx={{ flex: 1 }} />
            <CustomButton title="Search" size="medium" primary onClick={searchClick} sx={{ flex: 1 }} />
          </Box>
        </Box>

        <Box sx={{ mt: 4, pt: 3, borderTop: '1px solid rgba(15,23,42,0.1)', display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <CustomButton title="Select All" size="medium" onClick={selectAllClick} sx={{ flex: 1 }} />
            <CustomButton title="Deselect All" size="medium" onClick={deselectAllClick} sx={{ flex: 1 }} />
          </Box>
          <CustomButton disabled={isTrake} title={isGroup ? "Ungroup Results" : "Group by Video"} size="large" onClick={() => setIsGroup(!isGroup)} />
          <CustomTextField label="VQA Answer" inputRef={VQARef} multiline rows={2} />
          <CustomButton title="Save Output" size="large" primary onClick={handleSaveOutput} />
        </Box>
      </Box>

      {/* RIGHT MAIN AREA */}
      <Box sx={{ flex: 1, overflowX: 'hidden', overflowY: 'scroll', borderRadius: 'var(--radius-lg)', className: 'glass-panel', bgcolor: 'var(--glass-bg)', p: 3, boxShadow: 'var(--shadow-md)' }}>
        {!isTrake && !isGroup && (
          <Grid container spacing={2} sx={{ alignContent: 'flex-start' }}>
            {result.length > 0 ? result.map((frame) => (
              <MainFrame 
                key={frame} 
                frame={frame} 
                allOrder={allOrder} 
                isTrake={isTrake}
                onOpenDialogTrake={handleOpenDialogTrake}
                onOpenDialog={handleOpenDialog}
                onFrameClick={handleFrameClick}
              />
            )) : <NoResultDiv />}
          </Grid>
        )}

        {!isTrake && isGroup && (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {groupedRes.length > 0 ? groupedRes.map((resultGroup, idx) => (
              <Box key={idx} className="glass-panel" sx={{ p: 2, overflowX: 'scroll', bgcolor: '#ffffff' }}>
                <Grid container spacing={2} sx={{ flexWrap: 'nowrap', minWidth: 'fit-content' }}>
                  {resultGroup.map((frame) => (
                    <MainFrame 
                      key={frame} 
                      frame={frame} 
                      allOrder={allOrder}
                      isTrake={isTrake}
                      onOpenDialogTrake={handleOpenDialogTrake}
                      onOpenDialog={handleOpenDialog}
                      onFrameClick={handleFrameClick}
                    />
                  ))}
                </Grid>
              </Box>
            )) : <NoResultDiv />}
          </Box>
        )}

        {isTrake && (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {trakeRes.length > 0 ? trakeRes.map((resultGroup, groupIdx) => (
              <Box key={groupIdx} className="glass-panel" sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 2, overflowX: 'scroll', bgcolor: '#ffffff' }}>
                <Typography 
                  variant="body1" 
                  onClick={(e) => handleGroupClick(e, groupIdx)}
                  sx={{ 
                    fontWeight: 'bold', color: groupOrder.get(groupIdx) ? '#ffffff' : 'var(--primary-color)', 
                    bgcolor: groupOrder.get(groupIdx) ? 'var(--accent-color)' : 'rgba(15,23,42,0.05)', 
                    width: '40px', height: '40px', borderRadius: 'var(--radius-sm)', 
                    cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                    transition: 'all 0.2s', '&:hover': { bgcolor: groupOrder.get(groupIdx) ? 'var(--accent-color)' : 'rgba(15,23,42,0.1)' }
                  }} 
                >
                  {groupOrder.get(groupIdx) || ''}
                </Typography>
                <Grid container spacing={2} sx={{ flexWrap: 'nowrap', minWidth: 'fit-content' }}>
                  {resultGroup.map((frame, frameIdx) => (
                    <MainFrame 
                      key={frame.mainFrame} 
                      frame={frame.mainFrame} 
                      inGroup={true} 
                      group={frame.allFrames} 
                      groupIdx={frameIdx} 
                      videoIdx={groupIdx}
                      allOrder={allOrder}
                      isTrake={isTrake}
                      onOpenDialogTrake={handleOpenDialogTrake}
                      onOpenDialog={handleOpenDialog}
                      onFrameClick={handleFrameClick}
                    />
                  ))}
                </Grid>
              </Box>
            )) : <NoResultDiv />}
          </Box>
        )}
      </Box>

      {/* DIALOGS */}
      <BootstrapDialog onClose={handleCloseDialog} open={openDialog}>
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontWeight: 600 }}>
          {frameDialog}
          <IconButton onClick={handleCloseDialog}><CloseIcon /></IconButton>
        </DialogTitle>
        <DialogContent dividers sx={{ display: 'flex', gap: 3, height: '100%' }}>
          <Box sx={{ width: '45%', display: 'flex', alignItems: 'center', bgcolor: '#000', borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
            <CardMedia component="iframe" src={videoDialog} sx={{ width: '100%', aspectRatio: '16/9', border: 'none' }} allowFullScreen />
          </Box>
          <Box sx={{ width: '55%', overflowY: 'scroll', pr: 1 }}>
            <Grid container spacing={2}>
              {neighborFrames.map((frame) => (
                <ResultIframe 
                  key={frame} 
                  src={IMAGE_DIR + frame + '.jpg'} 
                  orderNumber={allOrder.get(frame)} 
                  frameName={frame} 
                  onNumberClick={handleFrameClick} 
                />
              ))}
            </Grid>
          </Box>
        </DialogContent>
      </BootstrapDialog>

      <BootstrapDialog onClose={handleCloseDialogTrake} open={openDialogTrake}>
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontWeight: 600 }}>
          {frameDialog}
          <IconButton onClick={handleCloseDialogTrake}><CloseIcon /></IconButton>
        </DialogTitle>
        <DialogContent dividers sx={{ display: 'flex', gap: 3, height: '100%' }}>
          <Box sx={{ width: '45%', display: 'flex', alignItems: 'center', bgcolor: '#000', borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
            <CardMedia component="iframe" src={videoDialog} sx={{ width: '100%', aspectRatio: '16/9', border: 'none' }} allowFullScreen />
          </Box>
          <Box sx={{ width: '55%', display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Box sx={{ height: '40%', overflowY: 'scroll', pr: 1, pb: 2, borderBottom: '1px solid rgba(15,23,42,0.1)' }}>
              <Typography variant="subtitle2" sx={{ mb: 1, color: 'var(--text-muted)' }}>Related Sequence Frames</Typography>
              <Grid container spacing={2}>
                {relatedFrames.map((frame) => (
                  <ResultIframe 
                    key={frame.image_id} 
                    src={IMAGE_DIR + frame.image_id + '.jpg'} 
                    frameName={frame.image_id} 
                    checked={frame.image_id === mainFrameDialog} 
                    onNumberClick={handleRelatedClick} 
                  />
                ))}
              </Grid>
            </Box>
            <Box sx={{ height: '60%', overflowY: 'scroll', pr: 1 }}>
              <Typography variant="subtitle2" sx={{ mb: 1, color: 'var(--text-muted)' }}>Neighboring Context</Typography>
              <Grid container spacing={2}>
                {neighborFrames.map((frame) => (
                  <ResultIframe 
                    key={frame} 
                    src={IMAGE_DIR + frame + '.jpg'} 
                    frameName={frame} 
                    checked={frame === mainFrameDialog} 
                    onNumberClick={handleRelatedClick} 
                  />
                ))}
              </Grid>
            </Box>
          </Box>
        </DialogContent>
      </BootstrapDialog>
    </Box>
  );
}

export default App;