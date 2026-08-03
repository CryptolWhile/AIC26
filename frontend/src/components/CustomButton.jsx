import Button from '@mui/material/Button';

const CustomButton = ({ title, primary, onClick, size, disabled = false, sx = {} }) => {
  return (
    <Button
      onClick={onClick}
      disabled={disabled}
      disableElevation
      variant={primary ? 'contained' : 'outlined'}
      sx={{
        width: size === 'large' ? '100%' : size === 'medium' ? 'auto' : 'auto',
        flex: size === 'medium' ? 1 : undefined,
        height: '42px',
        textTransform: 'none',
        borderRadius: 'var(--radius-md)',
        fontWeight: 600,
        fontFamily: "'Inter', sans-serif",
        letterSpacing: '0.3px',
        transition: 'all var(--transition-fast)',
        backgroundColor: primary ? 'var(--primary-color)' : 'transparent',
        color: primary ? '#ffffff' : 'var(--primary-color)',
        border: primary ? 'none' : '1px solid rgba(15, 23, 42, 0.15)',
        boxShadow: primary ? 'var(--shadow-md)' : 'none',
        '&:hover': {
          backgroundColor: primary ? 'var(--primary-hover)' : 'rgba(15, 23, 42, 0.04)',
          transform: 'translateY(-1px)',
          boxShadow: primary ? 'var(--shadow-lg)' : 'var(--shadow-sm)',
        },
        '&:active': {
          transform: 'translateY(0)',
        },
        '&.Mui-disabled': {
          backgroundColor: 'rgba(0,0,0,0.05)',
          color: 'rgba(0,0,0,0.25)',
          border: 'none',
          boxShadow: 'none',
        },
        ...sx
      }}
    >
      {title}
    </Button>
  );
};

export default CustomButton;