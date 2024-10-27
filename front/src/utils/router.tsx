import { createHashRouter } from 'react-router-dom'
import App from 'src/components/App/App'
import UploadData from 'src/components/UploadData/UploadData'
import DataTable from 'src/components/DataTable/DataTable'

export const router = createHashRouter([
  {
    path: '/',
    element: <App />,
    children: [
      {
        path: 'upload',
        element: <UploadData />,
      },
      {
        path: 'data',
        element: <DataTable />,
      },
    ],
  },
])
