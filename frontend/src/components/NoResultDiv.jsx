import { Box, Typography } from '@mui/material';
import SearchOffIcon from '@mui/icons-material/SearchOff';
const NoResultDiv = () => {
  return (
    <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" height="100%" width="100%">
      <SearchOffIcon fontSize="large" sx={{ color: 'white' }} />
      <Typography variant="h6" sx={{ marginTop: 2, color: 'white', fontWeight: 'bold' }}>
        No results found
      </Typography>
    </Box>
  );
};

export default NoResultDiv;