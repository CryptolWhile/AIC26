import TextField from '@mui/material/TextField';

const CustomTextField = ({ inputRef, ...props }) => {
  return (
    <TextField
      variant="outlined"
      inputRef={inputRef}
      {...props}
      sx={{
        ...props.sx,
        '& .MuiInputBase-root': {
          backgroundColor: 'var(--glass-bg)',
          backdropFilter: 'blur(8px)',
          fontSize: '14px',
          borderRadius: 'var(--radius-md)',
          transition: 'all var(--transition-fast)',
          boxShadow: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.02)',
        },
        '& .MuiInputBase-input': {
          color: 'var(--text-main)', // Ensure text is visible
          padding: props.multiline ? '12px 14px' : '0 14px',
          height: props.multiline ? 'auto' : '42px', 
          boxSizing: 'border-box',
        },
        '& .MuiInputBase-root:hover': {
          backgroundColor: '#ffffff',
        },
        '& .MuiInputBase-root.Mui-focused': {
          backgroundColor: '#ffffff',
          boxShadow: '0 0 0 3px rgba(79, 70, 229, 0.15)',
        },
        '& .MuiInputLabel-root': {
          fontSize: '14px',
          color: 'var(--text-muted)',
          transform: props.multiline ? 'translate(14px, 12px) scale(1)' : 'translate(14px, 11px) scale(1)',
        },
        '& .MuiInputLabel-shrink': {
          transform: 'translate(14px, -9px) scale(0.85)',
          color: 'var(--primary-color)',
          fontWeight: 500,
          backgroundColor: '#ffffff',
          padding: '0 4px',
          borderRadius: '4px',
        },
        '& .MuiOutlinedInput-notchedOutline': {
          borderColor: 'rgba(15, 23, 42, 0.1)',
          borderWidth: '1px',
          transition: 'all var(--transition-fast)',
        },
        '&:hover .MuiOutlinedInput-notchedOutline': {
          borderColor: 'rgba(15, 23, 42, 0.25)',
        },
        '& .MuiOutlinedInput-root.Mui-focused .MuiOutlinedInput-notchedOutline': {
          borderColor: 'var(--accent-color)',
          borderWidth: '1.5px',
        }
      }} 
    />
  );
};

export default CustomTextField;