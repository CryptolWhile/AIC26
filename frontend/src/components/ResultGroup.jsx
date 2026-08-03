import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import ResultIframe from './ResultIframe';
import React from 'react';
import { IMAGE_DIR, VIDEO_DIR } from '../config/appConfig';
import { Typography, Input, Dialog, DialogTitle, DialogActions, DialogContent, IconButton, Button, styled, CardMedia, Card } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useState } from 'react';
function getVideoName(path) {
  let videoName = path.split('/').slice(-3, -1).join('/');
  return videoName;
}
const BootstrapDialog = styled(Dialog)(({ theme }) => ({
  '& .MuiDialogContent-root': {
    padding: theme.spacing(2),
  },
  '& .MuiDialogActions-root': {
    padding: theme.spacing(1),
  },
  '& .MuiDialog-paper': {
    width: '80%',
    maxWidth: 'none',
    height: '80%',
    borderRadius: '16px'
  },
}));
const ResultGroup = React.memo(function ResultGroup({ resultW = 200, resultH = 150, inGroup = false, allSrc, orderNumber, onNumberClick }) {
  const [neighborFrames, setNeighborFrames] = useState([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [frameDialog, setFrameDialog] = useState('');
  function handleCloseDialog(e) {
    e.stopPropagation();
    setOpenDialog(false);
  }
  async function handleOpenDialog(filename) {
    setOpenDialog(true);
    setFrameDialog(filename);
    let videoName = await getVideoName(filename);
    try {
      let res = await fetch(`/${videoName}/frames.txt`);
      let resText = await res.text();
      let listFrame = resText.split("\n");
      let index = listFrame.indexOf(filename);
      if (index !== -1) {
        let start = Math.max(0, index - 10);
        let end = Math.min(listFrame.length, index + 11);
        setNeighborFrames(listFrame.slice(start, end));
      } else {
        setNeighborFrames([]);
      }
    } catch (err) {
      console.error("Lỗi fetch frames:", err);
    }

  };

  const SortableFrame = React.memo(function SortableFrame({ frame }) {
    const imageSrc = IMAGE_DIR + frame + '.jpg';
    return (
      <div onClick={() => handleOpenDialog(frame)}
      >
        <ResultIframe inGroup={inGroup} width={resultW} height={resultH} src={imageSrc} frameName={frame}
        />
      </div>
    );
  });
  return (
    <>
      <React.Fragment>
        <BootstrapDialog
          onClose={(e) => { handleCloseDialog(e); }}
          open={openDialog}
          sx={{ zIndex: 3000 }}
        >
          <DialogTitle sx={{ m: 0, p: 2 }}>
            {frameDialog}
          </DialogTitle>
          <IconButton
            onClick={(e) => { handleCloseDialog(e); }}
            sx={(theme) => ({
              position: 'absolute',
              right: 8,
              top: 8,
              color: theme.palette.grey[500],
            })}
          >
            <CloseIcon />
          </IconButton>
          <DialogContent dividers sx={{ display: 'flex', flexDirection: 'row', columnGap: 2 }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', width: '40%', alignItems: 'center', justifyContent: 'center' }}>
              <CardMedia component="iframe" width="560" height="315" src={"https://www.youtube.com/embed/Rzpw5WR7nAY?si=78k3I2-zkCMnKFwi&amp;start=255"} controls allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" />
            </Box>
            <Grid container spacing={'14px'} sx={{ width: '60%', height: '100%', overflowX: 'hidden', overflowY: 'scroll', alignContent: 'flex-start', justifyContent: 'center', marginLeft: 'auto', borderRadius: '16px' }}>
              {neighborFrames.length > 0 && neighborFrames.map((frame,) => (
                <ResultIframe
                  inGroup={inGroup}
                  key={frame}
                  width={resultW}
                  height={resultH}
                  src={IMAGE_DIR + frame + '.jpg'}
                  frameName={frame}
                />
              ))}
            </Grid>
          </DialogContent>``
        </BootstrapDialog>
      </React.Fragment>
      <Box sx={{ display: 'flex', flexDirection: 'row', alignItems: 'center', columnGap: 1, overflow: 'scroll' }}>
        {orderNumber && (<Typography
          variant="body1"
          sx={{
            fontWeight: 'bold',
            color: 'var(--primary-color)',
            backgroundColor: 'white',
            width: '30px',
            height: '40px',
            borderRadius: '4px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onClick={onNumberClick}
        >
          {orderNumber || ''} 
        </Typography>)}
        <Box sx={{ overflowX: 'scroll', backgroundColor: 'white', borderRadius: '16px' }}>

          <Grid container spacing={'14px'} sx={{ marginLeft: 'auto', padding: '10px', overflowX: 'scroll', flexWrap: 'nowrap', alignContent: 'flex-start', justifyContent: 'center', minWidth: 'fit-content' }}>
            {allSrc.map((frame) => (
              <SortableFrame key={frame} frame={frame} />
            ))}
          </Grid>
        </Box>
      </Box>
    </>
  );
});

export default ResultGroup;