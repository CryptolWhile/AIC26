import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import React, { useCallback } from 'react';

const ResultIframe = React.memo(function ResultIframe({ width = 200, height = 150, src, orderNumber, frameName, onNumberClick, inGroup, checked }) {
  const titleH = 32; // Fixed height for title bar for a cleaner look
  const imgH = height - titleH;

  const handleClick = useCallback((e) => {
    if (onNumberClick) onNumberClick(e, frameName);
  }, [onNumberClick, frameName]);

  return (
    <Grid className="content-visibility-auto" sx={{ width: `${width}px`, height: `${height}px`, flexShrink: 0, position: 'relative' }}>
      <Box 
        sx={{ 
          backgroundColor: '#ffffff', 
          borderRadius: 'var(--radius-md)', 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column', 
          boxShadow: 'var(--shadow-sm)',
          border: checked ? '2px solid var(--accent-color)' : '1px solid rgba(15, 23, 42, 0.08)',
          overflow: 'hidden',
          transition: 'transform var(--transition-fast), box-shadow var(--transition-fast)',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: 'var(--shadow-md)'
          }
        }}
      >
        <Box sx={{ display: 'flex', flexDirection: 'row', alignItems: 'center', height: `${titleH}px`, padding: '0 8px', gap: 1, backgroundColor: 'rgba(248, 250, 252, 0.8)', borderBottom: '1px solid rgba(15, 23, 42, 0.05)' }}>
          {!inGroup && (
            <Typography 
              variant="body2" 
              onClick={handleClick}
              sx={{
                fontWeight: 600, 
                color: (orderNumber || checked) ? '#ffffff' : 'var(--text-muted)', 
                backgroundColor: (orderNumber || checked) ? 'var(--accent-color)' : 'transparent', 
                border: (orderNumber || checked) ? 'none' : '1px solid rgba(15, 23, 42, 0.2)',
                width: '24px', 
                height: '24px', 
                borderRadius: '6px', 
                cursor: 'pointer', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                transition: 'all 0.2s',
                fontSize: '0.75rem',
                '&:hover': {
                  backgroundColor: (orderNumber || checked) ? 'var(--accent-color)' : 'rgba(15, 23, 42, 0.05)'
                }
              }} 
            >
              {orderNumber || ''}
              {(checked && !orderNumber) ? '✓' : ''}
            </Typography>
          )}
          <Typography 
            variant='caption' 
            sx={{ 
              fontWeight: 500,
              color: 'var(--text-main)',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap'
            }}
          >
            {frameName}
          </Typography>
        </Box>
        <Box
          sx={{
            width: '100%',
            height: `${imgH}px`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#f8fafc',
            position: 'relative'
          }}
        >
          <Box component="img"
            src={src}
            loading="lazy"
            decoding="async"
            sx={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
            }} 
          />
        </Box>
      </Box>
    </Grid>
  );
});

export default ResultIframe;