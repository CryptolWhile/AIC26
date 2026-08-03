import { useTheme } from '@mui/material/styles';
import Box from '@mui/material/Box';
import OutlinedInput from '@mui/material/OutlinedInput';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import Chip from '@mui/material/Chip';
import { models } from '../constants/options';

const ITEM_HEIGHT = 48;
const ITEM_PADDING_TOP = 8;
const MenuProps = {
  disablePortal: true,
  PaperProps: {
    style: {
      maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
      width: 250,
    },
  },
};

function getStyles(name, personName, theme) {
  return {
    fontWeight: personName.includes(name)
      ? theme.typography.fontWeightMedium
      : theme.typography.fontWeightRegular,
  };
}

export default function MultipleSelectChip({ selectedModel, setSelectedModel }) {
  const theme = useTheme();
  const handleChange = (event) => {
    const {
      target: { value },
    } = event;
    setSelectedModel(
      typeof value === 'string' ? value.split(',') : value,
    );
  };

  return (
    <div>
        <Select
          labelId="demo-multiple-chip-label"
          id="demo-multiple-chip"
          multiple
          value={selectedModel}
          onChange={handleChange}
          sx={{ width: '100%', borderRadius: '12px' }}
          input={
            <OutlinedInput id="select-multiple-chip" focused
              sx={{
                '& .MuiInputBase-root': {
                  backgroundColor: 'var(--glass-bg)',
                  backdropFilter: 'blur(8px)',
                  fontSize: '14px',
                  borderRadius: 'var(--radius-md)',
                  transition: 'all var(--transition-fast)',
                },
                '& .MuiInputBase-root:hover': {
                  backgroundColor: '#ffffff',
                },
                '& .MuiInputBase-root.Mui-focused': {
                  backgroundColor: '#ffffff',
                  boxShadow: '0 0 0 3px rgba(79, 70, 229, 0.15)',
                },
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(15, 23, 42, 0.1)',
                  borderWidth: '1px',
                  transition: 'all var(--transition-fast)',
                },
                '&:hover .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(15, 23, 42, 0.25)',
                },
                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'var(--accent-color)',
                  borderWidth: '1.5px',
                }
              }} />
          }
          renderValue={(selected) => (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, zIndex: 10000 }}>
              {selected.map((value) => (
                <Chip key={value} label={value} />
              ))}
            </Box>
          )}
          MenuProps={MenuProps}
        >
          {models.map((name) => (
            <MenuItem
              key={name}
              value={name}
              style={getStyles(name, selectedModel, theme)}
            >
              {name}
            </MenuItem>
          ))}
        </Select>
      </div>
  );
}